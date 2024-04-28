use std::collections::HashMap;
use kohonen::network::network::{prepare_dataset, Network};
use plotters::prelude::*;
use rand::Rng;

fn main() {
    let (dataset, normalized, names, stats) = prepare_dataset("Country_Dataset.csv");
    // let neuron_count = 5;
    // let mut network = Network::new(neuron_count, 9);
    // let res1 = network.inference(&normalized);
    // network.train(&normalized, 0.0, 0.75, 100, 0.3, 5.5);
    // let res2 = network.inference(&normalized);
    // println!("{:?}", count(&res1));
    // println!("{:?}", count(&res2));
    // println!("{:?}", count(&res2).len());

    // let mut colors = vec![];
    // for _ in 0..neuron_count {
    //     colors.push(generate_random_color());
    // }

    // for i in 0..9 {
    //     for j in (i + 1)..9 {
    //         let (points, min_x, max_x, min_y, max_y) = generate_points(&dataset, i, j, &res2);
    //         // _ = draw(format!("output/net/{}_{}.svg", names[i], names[j]).as_str(), format!("{}_{}",names[i], names[j]).as_str(), points, stats[i].0, stats[i].1, stats[j].0, stats[j].1, &colors);
    //         _ = draw(format!("output/net/{}_{}.svg", names[i], names[j]).as_str(), format!("{}_{}",names[i], names[j]).as_str(), points, min_x, max_x, min_y, max_y, &colors);   
    //     }
    // }

    let w = 8;
    let h = 5;
    let w_res = 600;
    let h_res = 600;
    let mut map = Network::new((w * h) as u8, 9);
    map.train_map(&normalized, 150., 0.75, 10000, 0.3, 5.5, w, h, w_res, h_res);
    let res = map.inference(&normalized);
    for i in 0..9 {
        _ = draw_map(format!("output/map/{}.svg", names[i]).as_str(), &prepare_data(&normalized, &res, i, w * h), w, h, w_res, h_res);
    }
}

fn draw(file_name: &str, name: &str, data: Vec<((f32, f32), u8)>, min_x: f32, max_x: f32, min_y: f32, max_y: f32, colors: &Vec<RGBColor>) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);
    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 40).into_font())
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|x| format!("{:}", x))
        .y_label_formatter(&|x| format!("{:}", x))
        .draw()?;

    chart.draw_series(data.iter().map(|(point, color_index)| Circle::new(*point, 2, colors[*color_index as usize])))?;
    root.present()?;

    Ok(())
}

fn draw_map(file_name: &str, gray_values: &Vec<f32>, w: usize, h: usize, w_res: u16, h_res: u16) -> Result<(), Box<dyn std::error::Error>>  {
    let root_drawing_area =
        SVGBackend::new(file_name, (w_res.into(), h_res.into())).into_drawing_area();
    let child_drawing_areas = root_drawing_area.split_evenly((w, h));
    for (area, &gray_value) in child_drawing_areas.into_iter().zip(gray_values.iter()) {
        let gray = (gray_value * 255.0) as u8;
        let gray_color = RGBColor(gray, gray, gray);
        area.fill(&gray_color)?;
    }
    root_drawing_area.present()?;
    Ok(())
}

fn prepare_data(data: &Vec<Vec<f32>>, clusters: &Vec<u8>, index: usize, neurons_count: usize) -> Vec<f32> {
    let mut tmp = vec![(0.0, 0); neurons_count];
    for i in 0..data.len() {
        tmp[clusters[i] as usize].0 += data[i][index];
        tmp[clusters[i] as usize].1 += 1;
    }
    
    let mut res = Vec::with_capacity(neurons_count);
    for elem in tmp {
        if elem.1 == 0 {
            res.push(0.0);
            continue;
        }
        res.push(elem.0 / elem.1 as f32);
    }
    res
}

fn count(vector: &Vec<u8>) -> HashMap<u8, u32> {
    let mut counts: HashMap<u8, u32> = HashMap::new();

    for element in vector {
        let count = counts.entry(*element).or_insert(0);
        *count += 1;
    }

    counts
}

fn generate_points(data: &Vec<Vec<f32>>, first_param: usize, second_param: usize, clusters: &Vec<u8>) -> (Vec<((f32, f32), u8)>, f32, f32, f32, f32) {
    let mut res: Vec<((f32, f32), u8)> = vec![];

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for i in 0..data.len() {
        let x = data[i][first_param];
        let y = data[i][second_param];

        min_x = f32::min(x, min_x);
        max_x = f32::max(x, max_x);
        min_y = f32::min(y, min_y);
        max_y = f32::max(y, max_y);

        res.push(((x, y), clusters[i]));
    }

    (res, min_x, max_x, min_y, max_y)
}

fn generate_random_color() -> RGBColor {
    let mut rng = rand::thread_rng();
    let mut r: u8;
    let mut g;
    let mut b;

    loop {
        r = rng.gen_range(0..=255);
        g = rng.gen_range(0..=255);
        b = rng.gen_range(0..=255);

        if r - g < 50 || g - b < 50 || r - b < 50 {
            continue;
        }

        break;
    }


    RGBColor(r, g, b)
}