use clap::Parser;
use optimist_router::experiment::config::Config;
use std::fs;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to experiment configuration in YAML.
    #[clap(long, required = true)]
    config: String,

    /// Path to the root directory where datasets are stored.
    #[clap(long, required = true)]
    root: String,
}

fn main() {
    let args = Args::parse();
    let configs = fs::read_to_string(args.config).expect("Failed to read the configuration file.");
    let mut configs: Vec<Config> =
        serde_yaml::from_str(&configs).expect("Failed to parse the configuration file.");

    println!("Updating paths with root {}.", args.root);
    configs
        .iter_mut()
        .for_each(|config| config.update_paths(args.root.as_str()));
    println!("Executing {} experiments.", configs.len());
    configs.iter().for_each(|config| config.execute());
}
