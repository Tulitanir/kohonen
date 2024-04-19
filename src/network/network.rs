use std::f32::consts::E;

use csv::Reader;

use super::neuron::{self, Neuron};

pub struct Network {
    neurons: Vec<Neuron>,
}

impl Network {
    pub fn new(neuron_count: u8, inputs_num: u8) -> Self {
        let mut neurons = vec![];
        for _ in 0..neuron_count {
            neurons.push(Neuron::new(inputs_num, neuron_count))
        }

        Network { neurons: neurons }
    }
}

impl Network {
    pub fn inference(&mut self, dataset_path: &str, learning_dist: f32, min_potential: f32, epochs_num: u16, a: f32, b: f32) {
        
        let (dataset, normalized, names) = prepare_dataset(dataset_path);
        
        
        self.train(&normalized, learning_dist, min_potential, epochs_num, a, b);

    }
}

impl Network {
    fn train(&mut self, dataset: &Vec<Vec<f32>>, learning_dist: f32, min_potential: f32, epochs_num: u16, a: f32, b: f32) {
        let neurons_num = self.neurons.len() as u8;
        for e in 0..epochs_num {
            let mut sum = 0.0;
            for j in 0..dataset.len() {
                let mut min_distance = f32::MAX;
                let mut winner_index = 0;
                for i in 0..self.neurons.len() {
                    let dist = calculate_distance(&dataset[j], &self.neurons[i].weights);
                    if dist < min_distance {
                        min_distance = dist;
                        winner_index = i;
                    }
                }
                let mut neurons_to_train: Vec<usize> = vec![winner_index];
                for i in 0..self.neurons.len() {
                    if i == winner_index {
                        continue;
                    }

                    let dist = calculate_distance(&self.neurons[winner_index].weights, &self.neurons[i].weights);
                    if dist < learning_dist {
                        neurons_to_train.push(i);
                    }
                }

                sum += sub_vectors(&dataset[j], &self.neurons[winner_index].weights).iter().sum::<f32>() / dataset[j].len() as f32;

                for index in &neurons_to_train {
                    if self.neurons[*index].get_potential() > min_potential {
                        self.neurons[*index].weights = sum_vectors(&self.neurons[*index].weights, 
                            &sub_vectors(&dataset[j], &self.neurons[*index].weights).iter()
                            .map(|x| x * coefficient_fun(min_distance, j, a, b)).collect());
                        self.neurons[*index].change_potential_winner(min_potential);
                    }

                    self.neurons[*index].change_potential(neurons_num);
                }

                for i in 0..self.neurons.len() {
                    if neurons_to_train.contains(&i) {
                        continue;
                    }

                    self.neurons[i].change_potential(neurons_num);
                }
            }

            println!("Ошибка на эпохе {}: {}", e, sum / dataset.len() as f32);
        }
    }
}

fn coefficient_fun(d: f32, k: usize, a: f32, b: f32) -> f32 {
    gauss_function(d, k) * learning_rate_func(k, a, b)
}

fn gauss_function(d: f32, k: usize) -> f32 {
    f32::powf(E, -(d / (2 * k) as f32))
}

fn learning_rate_func(k: usize, a: f32, b: f32) -> f32 {
    a / (k as f32 + b)
}

fn sum_vectors(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x + y).collect()
}

fn sub_vectors(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x - y).collect()
}

fn calculate_distance(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    let mut sum = 0.0;
    
    for i in 0..vec1.len() {
        sum += f32::powi(vec1[i] - vec2[i], 2);        
    }

    f32::sqrt(sum)
}

fn prepare_dataset(dataset_path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, [&str; 9]) {
    let result = Reader::from_path(dataset_path);

    if result.is_err() {
        println!("Не удалось открыть файл");
        std::process::exit(1);
    }

    let names = ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"];
    let mut dataset = vec![];
    let mut normalized = vec![];
    
    let mut reader = result.unwrap();

    let mut counter = 0;
    let mut stats = [(f32::MAX, f32::MIN); 9];
    for (i, record) in reader.records().enumerate() {
        let spec = record.unwrap();
        dataset.push(vec![]);
        for j in 1..10 {
            let tmp = spec.get(j).unwrap().parse::<f32>().unwrap();
            if tmp < stats[j - 1].0 {
                stats[j - 1].0 = tmp;
            }

            if tmp > stats[j - 1].1 {
                stats[j - 1].1 = tmp;
            }

            dataset[i].push(tmp);
        }
    }

    for i in 0..dataset.len() {
        normalized.push(vec![]);

        for j in 0..dataset[i].len() {
            normalized[i].push(normalization(dataset[i][j], stats[j].0, stats[j].1))
        }
    }

    (dataset, normalized, names)
}

fn normalization(x: f32, xmin: f32, xmax: f32) -> f32 {
    (x - xmin) / (xmax - xmin)
}