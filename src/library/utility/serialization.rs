pub trait FileSerializable {
    fn save_to_file(&self, file_location: &str);

    fn read_from_file(path: &str) -> Self;
}

#[macro_export]
macro_rules! implement_serialization {
    () => {
        fn save_to_file(&self, file_location: &str) {
            std::fs::create_dir_all(std::path::Path::new(file_location).parent().unwrap())
                .expect("Unable to create directories");
            let file = std::fs::File::create(file_location).expect("Failed to create file");
            if file_location.ends_with("gz") {
                let gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
                let writer = std::io::BufWriter::new(gz);
                bincode::serialize_into(writer, &self).expect("Failed to serialize");
            } else {
                let writer = std::io::BufWriter::new(file);
                bincode::serialize_into(writer, &self).expect("Failed to serialize");
            }
        }

        fn read_from_file(path: &str) -> Self {
            let file = std::fs::File::open(path).expect("Faled to open file");
            if path.ends_with("gz") {
                let reader = std::io::BufReader::new(flate2::read::GzDecoder::new(file));
                bincode::deserialize_from(reader).expect("Failed to deserialize")
            } else {
                let reader = std::io::BufReader::new(file);
                bincode::deserialize_from(reader).expect("Failed to deserialize")
            }
        }
    };
}
