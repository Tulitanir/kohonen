use std::{collections::HashMap, fmt::format};
use kohonen::network::network::{prepare_dataset, Network};
use plotters::{data, prelude::*};

fn main() {
    let neuron_count = 3;
    let mut network = Network::new(neuron_count, 9);
    let (dataset, normalized, names, stats) = prepare_dataset("Country_Dataset.csv");
    let res1 = network.inference(&normalized);
    network.train(&normalized, 0.0, 0.75, 1000, 3., 10.);
    let res2 = network.inference(&normalized);
    println!("{:?}", count(&res1));
    println!("{:?}", count(&res2));

    let points1 = generate_points(&dataset, res1);
    let points2 = generate_points(&dataset, res2);

    for i in 0..9 {
//        _ = draw(format!("output/before/{}.svg", names[i]).as_str(), names[i], points1[i].clone(), neuron_count as f32, stats[i].0, stats[i].1);
        _ = draw(format!("output/after/{}.svg", names[i]).as_str(), names[i], points2[i].clone(), neuron_count as f32, stats[i].0, stats[i].1);
    }

    let w = 10;
    let h = 5;
    let w_res = 600;
    let h_res = 600;
    let mut map = Network::new((w * h) as u8, 9);
    map.train_map(&normalized, 10., 0.75, 1000, 3., 10., w, h, w_res, h_res);
    let grey_values = prepare_data(&normalized);
    for i in 0..9 {
        _ = draw_map(format!("output/map/{}.svg", names[i]).as_str(), &grey_values[i], w, h, w_res, h_res);
    }
}

fn count(vector: &Vec<u8>) -> HashMap<u8, u32> {
    let mut counts: HashMap<u8, u32> = HashMap::new();

    for element in vector {
        let count = counts.entry(*element).or_insert(0);
        *count += 1;
    }

    counts
}

fn generate_points(data: &Vec<Vec<f32>>, clusters: Vec<u8>) -> Vec<Vec<(f32, f32)>> {
    let mut res: Vec<Vec<(f32, f32)>> = vec![];

    for i in 0..9 {
        let mut tmp = vec![];
        for j in 0..clusters.len() {
            tmp.push((0.0, 0.0));
        }
        res.push(tmp);
    }

    for i in 0..9 {
        for j in 0..clusters.len() {
            res[i][j] = (clusters[j] as f32 + 1.0, data[j][i]);
        }
    }

    res
}

fn draw(file_name: &str, name: &str, data: Vec<(f32, f32)>, clusters_num: f32, min: f32, max: f32) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE);
    let root = root.margin(10, 10, 10, 10);
    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 40).into_font())
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..clusters_num, min..max)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|x| format!("{:}", x))
        .y_labels(5)
        .draw()?;

    chart.draw_series(PointSeries::of_element(
        data,
        2,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
        },
    ))?;
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

fn prepare_data(data: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = data[0].len();
    let cols = data.len();

    let mut res = vec![];

    for i in 0..rows {
        let mut tmp = vec![];
        for j in 0..cols {
            tmp.push(data[j][i]);
        }
        res.push(tmp);
    }

    res
}