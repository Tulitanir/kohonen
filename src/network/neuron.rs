use rand::Rng;

pub struct Neuron {
    pub weights: Vec<f32>,
    potential: f32
}

impl Neuron {
    pub fn new (weights_num: u8, neurons_num: u8) -> Self {
        let mut weights = vec![];

        let mut rng = rand::thread_rng();

        for _i in 0..weights_num {
            weights.push(rng.gen());
        }

        Neuron { weights, potential: 1.0 / neurons_num as f32}
    }

    pub fn get_potential(&self) -> f32 {
        self.potential
    }

    pub fn change_potential(&mut self, neurons_num: u8) {
        self.potential += 1.0 / neurons_num as f32;
    }

    pub fn change_potential_winner(&mut self, min_potential: f32) {
        self.potential -= min_potential;
    }
}