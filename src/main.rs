use kohonen::network::network::Network;

fn main() {
    let mut network = Network::new(3, 9);
    network.inference("Country_Dataset.csv", 0., 0., 0, 0., 0.);
}
