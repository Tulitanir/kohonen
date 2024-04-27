use std::{f32::consts::E, usize};

use csv::Reader;
use rand::seq::index;

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
    pub fn inference(&mut self, dataset: &Vec<Vec<f32>>) -> Vec<u8> {
        let mut res =vec![];
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
            res.push(winner_index as u8);
        }

        res
    }
}

impl Network {
    pub fn train(&mut self, dataset: &Vec<Vec<f32>>, learning_dist: f32, min_potential: f32, epochs_num: u16, a: f32, b: f32) {
        let neurons_num = self.neurons.len() as u8;
        for e in 0..epochs_num {
            let mut sum = 0.0;
            for j in 0..dataset.len() {
                let mut min_distance = f32::MAX;
                let mut winner_index = 0;
                for i in 0..self.neurons.len() {
                    if self.neurons[i].get_potential() > min_potential {
                        let dist = calculate_distance(&dataset[j], &self.neurons[i].weights);
                        if dist < min_distance {
                            min_distance = dist;
                            winner_index = i;
                        }
                    }
                }
                let neurons_to_train: Vec<usize> = vec![winner_index];
                // for i in 0..self.neurons.len() {
                //     if i == winner_index {
                //         continue;
                //     }

                //     let dist = calculate_distance(&self.neurons[winner_index].weights, &self.neurons[i].weights);
                //     if dist < learning_dist && self.neurons[i].get_potential() > min_potential {
                //         neurons_to_train.push(i);
                //     }
                // }
                // println!("Neurons to train: {:?}", neurons_to_train);

                sum += f32::powi(sub_vectors(&dataset[j], &self.neurons[winner_index].weights).iter().sum::<f32>() / dataset[j].len() as f32, 2);

                for i in 0..neurons_to_train.len() {
                    let index = neurons_to_train[i];
                    // println!("Changing weights of neuron: {:?}", index);
                    let sub = sub_vectors(&dataset[j], &self.neurons[index].weights);

                    let mul_by = func(e, epochs_num);

                    self.neurons[index].weights = sum_vectors(&self.neurons[index].weights,&mult_by(&sub, mul_by));

                    self.neurons[index].change_potential_winner(min_potential);
                }

                for i in 0..self.neurons.len() {
                    if neurons_to_train.contains(&i) {
                        continue;
                    }

                    self.neurons[i].change_potential(neurons_num);
                }
            }

            if e % 100 == 0 {
                println!("Ошибка на эпохе {}: {}", e, sum / dataset.len() as f32);
            }

        }
    }

    pub fn train_map(&mut self, dataset: &Vec<Vec<f32>>, learning_dist: f32, min_potential: f32, epochs_num: u16, a: f32, b: f32, w: usize, h: usize, w_res: u16, h_res: u16) {
        let neurons_num = self.neurons.len() as u8;
        for e in 0..epochs_num {
            let mut sum = 0.0;
            for j in 0..dataset.len() {
                let mut min_distance = f32::MAX;
                let mut winner_index = 0;
                let mut f = false;
                for i in 0..self.neurons.len() {
                    if self.neurons[i].get_potential() > min_potential {
                        let dist = calculate_distance(&dataset[j], &self.neurons[i].weights);
                        if dist < min_distance {
                            min_distance = dist;
                            winner_index = i;
                            f = true;
                        }
                    }
                }
                let x_win = calc_w(winner_index, w, h, w_res);
                let y_win = calc_h(winner_index, h, h_res);
                let mut neurons_to_train: Vec<usize> = vec![];
                if f {
                    neurons_to_train.push(winner_index);

                    for i in 0..self.neurons.len() {
                        if i == winner_index {
                            continue;
                        }
                        let x = calc_w(i, w, h, w_res);
                        let y = calc_h(i, h, h_res);
                        let dist = calculate_distance(&vec![x_win, y_win], &vec![x, y]);
                        if dist < learning_dist && self.neurons[i].get_potential() > min_potential {
                            neurons_to_train.push(i);
                        }
                    }
                    println!("Neurons to train: {:?}", neurons_to_train);
    
                    sum += f32::powi(sub_vectors(&dataset[j], &self.neurons[winner_index].weights).iter().sum::<f32>() / dataset[j].len() as f32, 2);
    
                    let x_win = calc_w(neurons_to_train[0], w, h, w_res);
                    let y_win = calc_h(neurons_to_train[0], h, h_res);
    
                    for i in 0..neurons_to_train.len() {
                        let index = neurons_to_train[i];
                        // println!("Changing weights of neuron: {:?}", index);
                        let sub = sub_vectors(&dataset[j], &self.neurons[index].weights);
    
                        let x = calc_w(i, w, h, w_res);
                        let y = calc_h(i, h, h_res);
                        let dist = calculate_distance(&vec![x_win, y_win], &vec![x, y]);
    
                        let mul_by = coefficient_fun(dist, e, a, b, epochs_num);
    
                        self.neurons[index].weights = sum_vectors(&self.neurons[index].weights,&mult_by(&sub, mul_by));
    
                        self.neurons[index].change_potential_winner(min_potential);
                    }
                }
                

                for i in 0..self.neurons.len() {
                    if neurons_to_train.contains(&i) {
                        continue;
                    }

                    self.neurons[i].change_potential(neurons_num);
                }
            }

            if e % 100 == 0 {
                println!("Ошибка на эпохе {}: {}", e, sum / dataset.len() as f32);
            }

        }
    }
}

fn calc_w(index: usize, w: usize, h: usize, w_res: u16) -> f32 {
    let step = w_res as f32 / w as f32;
    (index / h + 1) as f32 * step - 0.5 * step
}

fn calc_h(index: usize, h: usize, h_res: u16) -> f32 {
    let step = h_res as f32 / h as f32;
    (index % h + 1) as f32 * step - 0.5 * step
}

fn coefficient_fun(d: f32, k: u16, a: f32, b: f32, e: u16) -> f32 {
    gauss_function(d, k, e) * learning_rate_func(k, a, b)
}

fn gauss_function(d: f32, k: u16, e: u16) -> f32 {
    f32::powf(E, -(d / (2.0 * func(k, e))))
}

fn func(k: u16, epoch_num: u16) -> f32 {
    (epoch_num as f32 - k as f32) / (epoch_num as f32 * 10.0)
}

fn learning_rate_func(k: u16, a: f32, b: f32) -> f32 {
    a / (k as f32 + b)
}

fn sum_vectors(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x + y).collect()
}

fn sub_vectors(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x - y).collect()
}

fn mult_by(vec: &Vec<f32>, mul_by: f32) -> Vec<f32> {
    let mut res = Vec::with_capacity(vec.len());

    for elem in vec {
        res.push(elem * mul_by);
    }

    res
}

fn calculate_distance(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    let mut sum = 0.0;
    
    for i in 0..vec1.len() {
        sum += f32::powi(vec1[i] - vec2[i], 2);        
    }

    f32::sqrt(sum)
}

pub fn prepare_dataset(dataset_path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, [&str; 9], [(f32, f32); 9]) {
    let result = Reader::from_path(dataset_path);

    if result.is_err() {
        println!("Не удалось открыть файл");
        std::process::exit(1);
    }

    let names = ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"];
    let mut dataset = vec![];
    let mut normalized = vec![];
    
    let mut reader = result.unwrap();

    let mut stats = [(f32::MAX, f32::MIN); 9];
    let mut sums = [0.0; 9];
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

            sums[j - 1] += f64::powi(tmp as f64, 2);

            dataset[i].push(tmp);
        }
    }

    for i in 0..dataset.len() {
        normalized.push(vec![]);

        for j in 0..dataset[i].len() {
            normalized[i].push(normalization(dataset[i][j], f64::sqrt(sums[j]) as f32));
        }
    }

    (dataset, normalized, names, stats)
}

fn normalization(x: f32, div: f32) -> f32 {
    x / div 
}