use clap::ArgMatches;

use std::{
    io,
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
    text::{Span, Spans},
    widgets::{Block, Borders, List, ListItem, ListState},
    Frame, Terminal,
};

type BackendTerm = Terminal<CrosstermBackend<io::Stdout>>;

pub struct Tui {
    // --- [ Tui Backend ] ---
    terminal: BackendTerm,
    tickrate: Duration,
}

impl Tui {
    pub fn new(terminal: BackendTerm) -> Self {
        Tui {
            terminal,
            tickrate: Duration::from_millis(50),
        }
    }

    fn ui(f: &mut Frame<CrosstermBackend<io::Stdout>>) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(f.size());

        let pseudo_chunk = Layout::default().direction(Direction::Vertical).constraints([Constraint::Percentage(50), Constraint::Percentage(50)]).split(chunks[0]);

        let mut items: Vec<ListItem> = Vec::new();
        for i in 0..5 {
            let mut lines = vec![Spans::from("hels")];
            for _ in 0..1 {
                lines.push(Spans::from(Span::styled(
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    Style::default().add_modifier(Modifier::ITALIC),
                )));
            }
            let v = ListItem::new(lines).style(Style::default().fg(Color::Black).bg(Color::White));
            items.push(v);
        }



        let items = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("List"))
            .highlight_style(
                Style::default()
                    .bg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        let block = Block::default();
        f.render_widget(items, pseudo_chunk[0]);
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut last_tick = Instant::now();

        loop {
            self.terminal.draw(|f| Self::ui(f))?;

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

pub fn train_tui(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;

    let mut tui_app = Tui::new(terminal);

    tui_app.run()?;

    disable_raw_mode()?;
    execute!(
        tui_app.terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    tui_app.terminal.show_cursor()?;

    Ok(())
}
