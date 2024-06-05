use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering::Equal;

pub fn softmax(x: ArrayView1<f32>, temperature: Option<f32>) -> Array1<f32> {
    let temperature = temperature.unwrap_or(1_f32);
    let max = *x
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
        .unwrap();
    let x = x.mapv(|value| ((value - max) / temperature).exp());
    let sum = x.sum();
    x.mapv(|value| value / sum)
}

pub fn inverse_upper_triangular(matrix: ArrayView2<f32>) -> Array2<f32> {
    // This is the matrix D.
    let diag = Array2::<f32>::eye(matrix.ncols()) * matrix.diag();
    // This is U, a strictly upper triangular matrix.
    let upper = &matrix - &diag;

    // This is D^-1
    let diag_inverse = diag
        .to_owned()
        .mapv(|x| if x != 0_f32 { 1_f32 / x } else { 0_f32 });

    // Compute (I + D^-1 U)^-1 = \sum_{i=0}^{d-1} (-D^-1 U)^i.
    let unit = -diag_inverse.dot(&upper);
    let mut sum = Array2::<f32>::eye(matrix.ncols()) + &unit;
    let mut product = unit.to_owned();
    (2..matrix.ncols()).for_each(|_| {
        product = product.dot(&unit);
        sum += &product;
    });

    sum.dot(&diag_inverse)
}
