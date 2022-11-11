extern crate prost_build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(&["src/solvers.proto"],
                                &["src/"])?;
                                
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-lib={}=openblas", "dylib");
    println!("cargo:rustc-link-lib={}=cblas", "dylib");

    Ok(())
}