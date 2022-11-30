use crate::solvers::*;

#[macro_export]
macro_rules! with_open_cfg_network {
    ($var_name:ident, $path_cfg:expr, $do:block) => {
        let solver_type = solver_type_from_file($path_cfg)?;

        match solver_type.as_str() {
            "rmsprop" => {
                let mut solver = SolverRMS::from_file($path_cfg)?;
                // solver.load_state(solver_state)?;
                let $var_name = Network::new_for_test(solver, 10);
                $do
            }
            "sgd" => {
                let mut solver = SolverSGD::from_file($path_cfg)?;
                
                let $var_name = Network::new_for_test(solver, 10);
                $do
            }
            _ => {
                todo!()
            }
        }
    };
}