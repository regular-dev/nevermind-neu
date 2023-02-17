use clap::{Arg, ArgMatches};
use ndarray::Data;
use nevermind_neu::dataloader::*;
use nevermind_neu::err::*;
use nevermind_neu::network::*;
use nevermind_neu::{
    dataloader::ProtobufDataLoader, models::Sequential, network::CallbackReturnAction,
};

use log::{error, info};

use std::thread::JoinHandle;
use std::{
    error::Error,
    io,
    sync::mpsc::{channel, Receiver},
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Corner, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, ListState},
    Frame, Terminal,
};

use crate::train;
use crate::train::create_net_from_cmd_args;

const DATA: [(f64, f64); 5] = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];

pub enum NetMsg {
    InitInfo(usize, usize),    // epoch_size, batch_size
    StepInfo(usize, f32, f32), // iter, loss, accuracy
    // TODO : split message to enum's entries
    Stop,
}

type BackendTerm = Terminal<CrosstermBackend<io::Stdout>>;
type InfoVec = Vec<(usize, usize, f64, f64)>; // (epoch, iter, sum_error, accuracy)

pub struct Tui {
    // --- [ Tui Backend ] ---
    terminal: BackendTerm,
    tickrate: Duration,
    epoch_size: usize,
    batch_size: usize,
    net_recver: Receiver<NetMsg>,
    info_storage: InfoVec,
    // --- [ Plot params ] --- //
    pub epoch_bound: f64, // show last N epoch on the loss chart
}

impl Tui {
    pub fn new(
        terminal: BackendTerm,
        epoch_size: usize,
        batch_size: usize,
        net_recver: Receiver<NetMsg>,
    ) -> Self {
        Tui {
            terminal,
            tickrate: Duration::from_millis(100),
            epoch_size,
            batch_size,
            info_storage: Vec::new(),
            net_recver,
            epoch_bound: 15.0,
        }
    }

    fn max_in_vec(v: &Vec<(f64, f64)>) -> Option<f64> {
        if v.is_empty() {
            return None;
        }

        let mut max_val: f64 = v.first().unwrap().1;

        for (_, loss) in v.iter() {
            if *loss > max_val {
                max_val = *loss;
            }
        }

        return Some(max_val);
    }

    fn ui(
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        data: &InfoVec,
        epoch_size: usize,
        batch_size: usize,
        epoch_bound: f64,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(f.size());

        // let pseudo_chunk_top = Layout::default()
        //     .direction(Direction::Vertical)
        //     .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        //     .split(chunks[0]);

        let pseudo_chunk_right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        let mut items: Vec<ListItem> = Vec::with_capacity(data.len());

        for i in data.iter() {
            let done = (i.1 * batch_size) as f32 % epoch_size as f32 / epoch_size as f32;
            let avg_err = i.2 / (i.1 % (epoch_size / batch_size) ) as f64;
            let avg_acc = i.3 / (i.1 % (epoch_size / batch_size) ) as f64;

            let lines = vec![Spans::from(format!(
                "Epoch {} | Done {:.3}% | Error {:.3} | Accuracy {:.3}%",
                i.0, done, avg_err, avg_acc
            ))];

            items.push(ListItem::new(lines));
        }

        let items = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Training info"),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        let mut data_err_vec = Vec::with_capacity(data.len());
        let mut data_acc_vec = Vec::with_capacity(data.len());
        for i in data.iter() {
            data_err_vec.push((i.0 as f64, i.2 / (i.1 % (epoch_size / batch_size)) as f64));
            data_acc_vec.push((i.0 as f64, i.3 / (i.1 % (epoch_size / batch_size)) as f64));
        }

        let max_err_val = match Tui::max_in_vec(&data_err_vec) {
            Some(x) => x,
            None => 10.0,
        };

        let err_graph_data = vec![Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Yellow))
            .graph_type(GraphType::Line)
            .data(&data_err_vec.as_slice())];

        let acc_graph_data = vec![Dataset::default()
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::LightBlue))
            .graph_type(GraphType::Line)
            .data(&data_acc_vec.as_slice())];

        let err_chart = Chart::new(err_graph_data)
            .block(
                Block::default()
                    .title(Span::styled(
                        "Loss graph",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, epoch_bound])
                    .labels(vec![
                        Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("5", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("10", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("15", Style::default().add_modifier(Modifier::BOLD)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, max_err_val])
                    .labels(vec![
                        Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled(
                            format!("{:.3}", max_err_val),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                    ]),
            );

        let acc_chart = Chart::new(acc_graph_data)
            .block(
                Block::default()
                    .title(Span::styled(
                        "Accuracy graph",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, epoch_bound])
                    .labels(vec![
                        Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("5", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("10", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled("15", Style::default().add_modifier(Modifier::BOLD)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Accuracy")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 1.0])
                    .labels(vec![
                        Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled(
                            "1",
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                    ]),
            );

        f.render_widget(items, chunks[0]);
        f.render_widget(err_chart, pseudo_chunk_right[0]);
        f.render_widget(acc_chart, pseudo_chunk_right[1]);
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut last_tick = Instant::now();

        loop {
            // Example : "Epoch 0 | Done 43% | Error 0.415 | Accuracy 0.0%"
            while let Ok(m) = self.net_recver.try_recv() {
                if let NetMsg::StepInfo(iter_num, err, acc) = m {
                    let cur_epoch = (iter_num * self.batch_size) / self.epoch_size;

                    while self.info_storage.len() <= cur_epoch {
                        self.info_storage.push((cur_epoch, 0, 0.0, 0.0));
                    }

                    self.info_storage[cur_epoch].1 = iter_num;
                    self.info_storage[cur_epoch].2 += err as f64;
                    self.info_storage[cur_epoch].3 += acc as f64;
                }
            }

            self.terminal.draw(|f| {
                Self::ui(
                    f,
                    &self.info_storage,
                    self.epoch_size,
                    self.batch_size,
                    self.epoch_bound,
                )
            })?;

            let timeout = self
                .tickrate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        // KeyCode::Left => app.items.unselect(),
                        // KeyCode::Down => app.items.next(),
                        // KeyCode::Up => app.items.previous(),
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= self.tickrate {
                // app.on_tick();
                last_tick = Instant::now();
            }
        }

        Ok(())
    }
}

fn create_net_thread(
    args: &ArgMatches,
) -> Result<
    (
        Receiver<NetMsg>,
        JoinHandle<Result<(), Box<dyn Error + Send>>>,
    ),
    Box<dyn Error>,
> {
    let (sender, recver) = channel();
    let args_cloned = args.clone();

    let join_handle = thread::spawn(move || -> Result<(), Box<dyn Error + Send>> {
        let mut net = create_net_from_cmd_args(&args_cloned).unwrap();

        let epoch_size = net.train_dl.as_ref().unwrap().len().unwrap();
        let train_batch_size = net.train_batch_size().unwrap();

        // we need to send epoch_size and batch_size to ui thread
        // sender.send((epoch_size, train_batch_size as f32)).unwrap();
        sender.send(NetMsg::InitInfo(epoch_size, train_batch_size));

        net.add_callback(Box::new(
            move |iter_num, loss, accuracy| -> CallbackReturnAction {
                sender
                    .send(NetMsg::StepInfo(iter_num, loss, accuracy))
                    .unwrap();
                CallbackReturnAction::None
            },
        ));

        let mut opt_err = None;
        let mut opt_max_iter = None;

        if let Some(err) = args_cloned.get_one::<f32>("Err") {
            opt_err = Some(err);
        }

        if let Some(max_iter) = args_cloned.get_one::<usize>("MaxIter") {
            opt_max_iter = Some(max_iter);
        }

        let now_time = Instant::now();

        if opt_err.is_some() && opt_max_iter.is_some() {
            let err = opt_err.unwrap();
            let max_iter = opt_max_iter.unwrap();

            net.train_for_error_or_iter(*err, *max_iter);
        } else if opt_err.is_some() {
            let err = opt_err.unwrap();
            net.train_for_error(*err);
        } else if opt_max_iter.is_some() {
            let max_iter = opt_max_iter.unwrap();
            net.train_for_n_times(*max_iter);
        } else {
            return Err(Box::new(CustomError::WrongArg));
        }

        let elapsed = now_time.elapsed();

        info!("Elapsed for training : {} ms", elapsed.as_millis());

        Ok(())
    });

    Ok((recver, join_handle))
}

pub fn train_tui(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let (recver, join_handle) = create_net_thread(args)?;

    let init_msg = match recver.recv().unwrap() {
        NetMsg::InitInfo(epoch_size, batch_size) => (epoch_size, batch_size),
        _ => panic!("Received invalid NetMsg::InitInfo"),
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;

    let mut tui_app = Tui::new(terminal, init_msg.0, init_msg.1, recver);

    tui_app.run()?;

    disable_raw_mode()?;
    execute!(
        tui_app.terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    tui_app.terminal.show_cursor()?;

    join_handle.join().expect("Failed to join learning thread!");

    Ok(())
}
