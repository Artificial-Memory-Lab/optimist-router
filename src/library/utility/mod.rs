use ndarray::{Array2, ArrayView2, Axis};
use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;

pub mod linalg;
pub mod search_result;
pub mod serialization;

/// creates a progress bar with the default template
pub fn create_progress(name: &str, elems: usize) -> indicatif::ProgressBar {
    let pb = indicatif::ProgressBar::new(elems as u64);
    let rest =
        "[{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta}, SPEED: {per_sec})";
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(&format!("{}: {}", name, rest))
            .expect("Failed to create a progress bar."),
    );
    pb
}

pub fn random_split_ndarray<T: Clone>(
    seed: Option<u64>,
    data: ArrayView2<T>,
    proportion: f32,
) -> (Array2<T>, Array2<T>) {
    assert!(proportion > 0_f32 && proportion < 1_f32);

    let mut rng = match seed {
        None => StdRng::seed_from_u64(0),
        Some(seed) => StdRng::seed_from_u64(seed),
    };

    let mut indexes: Vec<usize> = (0..data.nrows()).collect();
    indexes.shuffle(&mut rng);

    let size = (proportion * data.nrows() as f32).ceil() as usize;

    (
        data.select(Axis(0), &indexes[..size]),
        data.select(Axis(0), &indexes[size..]),
    )
}

pub fn random_split_vector<T: Clone>(
    seed: Option<u64>,
    data: &[T],
    proportion: f32,
) -> (Vec<T>, Vec<T>) {
    assert!(proportion > 0_f32 && proportion < 1_f32);

    let mut rng = match seed {
        None => StdRng::seed_from_u64(0),
        Some(seed) => StdRng::seed_from_u64(seed),
    };

    let mut indexes: Vec<usize> = (0..data.len()).collect();
    indexes.shuffle(&mut rng);

    let size = (proportion * data.len() as f32).ceil() as usize;
    let split_a = indexes[..size]
        .iter()
        .map(|&i| data[i].to_owned())
        .collect::<Vec<_>>();
    let split_b = indexes[size..]
        .iter()
        .map(|&i| data[i].to_owned())
        .collect::<Vec<_>>();

    (split_a, split_b)
}
