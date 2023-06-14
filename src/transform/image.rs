use image::GenericImageView;

pub fn image_to_vector(image_path: &str) -> Vec<f64> {
    let image = image::open(image_path).expect("Cannot load");
    let resized_image = image.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);

    let normalized_image = resized_image.to_rgb8();
    let pixels = normalized_image.pixels();
    let mut vector = Vec::new();

    for pixel in pixels {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;

        vector.push(r);
        vector.push(g);
        vector.push(b);
    }

    vector
}