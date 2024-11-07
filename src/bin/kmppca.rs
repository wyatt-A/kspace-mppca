use core::ops::Range;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;
use cfl::ndarray::parallel::prelude::*;
use cfl::{num_complex, CflReader, CflWriter};
use cfl::{ndarray::ShapeBuilder, num_complex::Complex32};
use cfl::ndarray::{Array1, Array2};
use cs_table::ViewTable;
use kspace_mppca::{extract_data, insert_data, singular_value_threshold_mppca};
use mr_data::kspace::KSpace;
use rand_distr::ChiSquaredError;
use headfile::headfile::Headfile;
use serde::{Deserialize, Serialize};
use kspace_mppca::mp_denoise_matrix;
//use recon2::{config::config::TomlConfig, object_manager::object_manager::ObjectManager};


fn main() {

}
// fn main() {

//     let work_dir = "/privateShares/wa41/denoise_test2";

//     let object_manager = ObjectManager::from_file("/privateShares/wa41/24.chdi.01.work/240725-12-1/S69963/object-data/object-manager").unwrap();
//     let total_objs = object_manager.total_objects().unwrap();

//     println!("total objects: {}",total_objs);

//     // extract view coordinates
//     let k = object_manager.kspace(0).expect("failed to get kspace 0");

//     if k.len() > 0 {
//         println!("warning! Only using first kspace object!");
//     }

//     let phase_encoding_coords = k[0].coords();
//     let n_views = phase_encoding_coords.len();
//     let nx = k[0].line_len();
//     println!("nx: {}",nx);
//     println!("n views: {}",n_views);

//     let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
//         kx*kx + ky*ky + kz*kz
//     };

//     // let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
//     //     kx.abs().max(ky.abs()).max(kz.abs())
//     // };

//     let mut sample_ordering = vec![];
//     let mut idx = 0;
//     phase_encoding_coords.iter().for_each(|phase_encode|{
//         for kx in k_range(nx) {
//             let l2 = norm(kx,phase_encode[0],phase_encode[1]);
//             sample_ordering.push(
//                 SampleOrdering {linear_idx:idx,norm:l2}
//             );
//             idx += 1;
//         }
//     });
//     let n_samples_per_vol = idx;
//     println!("n_samples_per_vol: {}",n_samples_per_vol);
//     let chunk_size = 1000;

//     sample_ordering.sort_by_key(|order|order.norm);
    
//     let mut flattened = vec![];
//     let mut denoised = vec![];
//     let mut outputs = vec![];


//     // encode
//     for i in 1..11 {
//         println!("{}",i);
//         let ksp = object_manager.kspace(i).unwrap().remove(0);
//         //let ksp = KSpace::from_file(format!("k{:02}",i)).unwrap();
//         let c = ksp.to_array2(&phase_encoding_coords).into_dyn();
//         let mut sorted = Array1::<Complex32>::from_elem(c.len().f(), Complex32::ZERO);
//         let c_slice = c.as_slice_memory_order().unwrap();
//         sorted.iter_mut().zip(sample_ordering.iter()).for_each(|(elem,order)|{
//             //println!("r.0 = {}",r.0);
//             *elem = c_slice[order.linear_idx]
//         });

//         let p = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"r"));
//         let d = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"d"));
//         let o = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"k"));

//         cfl::from_array(
//             &p,
//             &sorted.into_dyn()
//         ).unwrap();

//         flattened.push(p);
//         denoised.push(d);
//         outputs.push(o);

//     }

//     // denoise
//     // let skip_indices = [0, 11, 22, 33, 44, 55]; // Indices to skip

//     // let filtered_range: Vec<usize> = (0..67)
//     //     .filter(|x| !skip_indices.contains(x)) // Filter out the unwanted indices
//     //     .collect();

//     let readers:Vec<_> = flattened.iter().map(|file|{
//         CflReader::new(file).unwrap()
//     }).collect();

//     let mut writers:Vec<_> = denoised.iter().map(|file|{
//         CflWriter::new(file,&[n_samples_per_vol]).unwrap()
//     }).collect();
 
//     //let mut init_energy = CflWriter::new("init_energy",&[16200]).unwrap();
//     //let mut final_energy = CflWriter::new("final_energy",&[16200]).unwrap();

//     let n_chunks = ceiling_div(16200*590, chunk_size);

//     let mut singular_values = CflWriter::new("singular_values",&[n_chunks]).unwrap();

//     //let mut idx:Vec<usize> = (0..590).collect();
//     //let mut sv_idx:Vec<usize> = (0..61).collect();
 
//     let indices:Vec<usize> = (0..n_samples_per_vol).collect();
    
//     for (i,chunk) in indices.chunks(chunk_size).enumerate() {
//         println!("working on {}",i);
//         let mut c_mat = extract_data(&chunk, &readers);
//         //println!("shape: {:?}",c_mat.shape());
//         //let _ = singular_value_threshold_mppca(&mut c_mat, None);
//         let info = singular_value_threshold_mppca(&mut c_mat, None);
//         let s_val = info.singular_values.get(0).unwrap();

//         singular_values.write(i, *s_val).unwrap();

//         insert_data(&chunk, &mut writers, &c_mat);
//         //idx.iter_mut().for_each(|x| *x += 590);
//         //sv_idx.iter_mut().for_each(|x| *x += 61);
//     }
 
//     // println!("flushing writers ...");
//     // for w in writers {
//     //     w.flush().expect("failed to flush writer");
//     // }

//     for (output,den) in outputs.iter().zip(denoised.iter()) {
//         println!("reconstructing {}",den.display());
//         let denoised = cfl::to_array(den, true).unwrap();
//         let mut compressed = Array2::from_elem((nx,n_views).f(), Complex32::ZERO);
//         let c = compressed.as_slice_memory_order_mut().unwrap();
//         denoised.iter().zip(sample_ordering.iter()).for_each(|(x,r)|{
//             c[r.linear_idx] = *x;
//         });
//         let ksp = KSpace::from_array2(compressed, &phase_encoding_coords);
//         ksp.write_to_file(output).unwrap();
//     }

//     // decode

// }


//cargo test --release --package kspace-mppca --bin kmppca -- resolve_input --exact --nocapture 
#[test]
fn prepare_input() {
    
    let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
    let n_vols = 67;
    let work_dir = kspace_dir.parent().unwrap().join("kmppca");


    create_dir_all(&work_dir).expect("failed to create work dir");
    let kspace_files:Vec<_> = (0..n_vols).map(|i|kspace_dir.join(format!("k{}",i)).join("k0")).collect();
    // load first kspace to get the common phase encoding coordinates
    let k = KSpace::from_file(&kspace_files[0]).expect("failed to load kspace data");
    let phase_encoding_coords_raw = k.coords();
    kspace_files.par_iter().for_each(|file|{
        let ksp = KSpace::from_file(file).expect("failed to load raw kspace data");
        let a = ksp.to_array2(&phase_encoding_coords_raw).into_dyn();
        let compressed_kspace_file = work_dir.join(file.parent().unwrap().file_name().unwrap());
        cfl::from_array(compressed_kspace_file, &a).unwrap();
    });
    ViewTable::from_coord_pairs(&phase_encoding_coords_raw).unwrap().write(work_dir.join("views")).unwrap();
}

//cargo test --release --package kspace-mppca --bin kmppca -- parse_headfiles --exact --nocapture 
#[test]
fn parse_headfiles() {
    let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
    let n_vols = 67;

    let headfiles:Vec<_> = (0..n_vols).map(|i|kspace_dir.join(format!("{}.headfile",i))).collect();

    let bval_key = "bvalue";
    let b_values:Vec<_> = headfiles.par_iter().map(|file|{
        let hf = Headfile::open(file).to_hash();
        let bval = hf.get(bval_key)
        .expect(&format!("failed to get {} from headfile",bval_key));
        bval.parse::<f32>().expect("failed to parse b-value to float")
    }).collect();

    println!("bvals: {:#?}",b_values);

    let max_bval = *b_values.iter().max_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();
    let min_bval = *b_values.iter().min_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();

    let b0_indices = [0,11,22,33,44,55];

}


//cargo test --release --package kspace-mppca --bin kmppca -- flatten --exact --nocapture
#[test]
fn flatten() {

    let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
    let n_vols = 67;
    let nx = 590;
    let work_dir = kspace_dir.parent().unwrap().join("kmppca");

    let phase_encoding_coords = ViewTable::from_file(work_dir.join("views"))
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    // l2 norm
    let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
        kx*kx + ky*ky + kz*kz
    };

    // l1 norm
    // let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
    //     kx.abs() + ky.abs() + kz.abs()
    // };

    // infinity norm
    // let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
    //     kx.abs().max(ky.abs()).max(kz.abs())
    // };

    // determine sample ordering based on some norm of k(r)
    let mut sample_ordering = vec![];
    let mut idx = 0;
    phase_encoding_coords.iter().for_each(|phase_encode|{
        for kx in k_range(nx) {
            let l2 = norm(kx,phase_encode[0],phase_encode[1]);
            sample_ordering.push(
                SampleOrdering {linear_idx:idx,norm:l2}
            );
            idx += 1;
        }
    });

    sample_ordering.sort_by_key(|x|x.norm);

    let mut ordering_fle = File::create(work_dir.join("sample_ordering")).unwrap();
    ordering_fle.write_all(
        &bincode::serialize(&sample_ordering).unwrap()
    ).unwrap();

    // sort k-space samples by their coordinate norm and write to 1-D cfl file
    (0..n_vols).into_par_iter().for_each(|vol_idx|{
        let compressed_kspace_file = work_dir.join(format!("k{vol_idx}"));
        let a = cfl::to_array(compressed_kspace_file, true).unwrap();
        let a_slice = a.as_slice_memory_order().unwrap();
        let mut sorted = Array1::<Complex32>::from_elem(a.len().f(), Complex32::ZERO);
        sorted.as_slice_memory_order_mut().unwrap()
        .par_iter_mut()
        .zip(sample_ordering.par_iter())
        .for_each(|(elem,order)|{
            *elem = a_slice[order.linear_idx]
        });
        cfl::from_array(work_dir.join(format!("f{vol_idx}")), &sorted.into_dyn()).unwrap();
    });

}

//cargo test --release --package kspace-mppca --bin kmppca -- unflatten --exact --nocapture
#[test]
fn unflatten() {

    let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
    let n_vols = 67;
    let nx = 590;
    let grid_size = [590,360,360];
    let work_dir = kspace_dir.parent().unwrap().join("kmppca");

    let phase_encoding_coords = ViewTable::from_file(work_dir.join("views"))
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    let mut ordering_fle = File::open(work_dir.join("sample_ordering")).unwrap();
    let mut bytes:Vec<u8> = vec![];
    ordering_fle.read_to_end(
        &mut bytes
    ).unwrap();

    let sample_ordering:Vec<SampleOrdering> = bincode::deserialize(&bytes).unwrap();

    // sort k-space samples by their coordinate norm and write to 1-D cfl file
    (0..n_vols).into_par_iter().for_each(|vol_idx|{

        let mut recon = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
        let r_slice = recon.as_slice_memory_order_mut().unwrap();

        let flattened = cfl::to_array(work_dir.join(format!("fd{vol_idx}")),true).unwrap();

        flattened.as_slice_memory_order().unwrap()
        .iter()
        .zip(sample_ordering.iter())
        .for_each(|(elem,order)|{
            r_slice[order.linear_idx] = *elem;
        });
        
        let gridded = KSpace::from_array2(recon,&phase_encoding_coords).grid(&grid_size);
        cfl::from_array(work_dir.join(format!("o{vol_idx}")), &gridded.into_dyn()).unwrap();

    });

    let rank = cfl::to_array(work_dir.join(format!("rank")),true).unwrap();
    let mut rank_recon = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
    let r_slice = rank_recon.as_slice_memory_order_mut().unwrap();
    rank.as_slice_memory_order().unwrap()
    .iter()
    .zip(sample_ordering.iter())
    .for_each(|(elem,order)|{
        r_slice[order.linear_idx] = *elem;
    });
    let gridded = KSpace::from_array2(rank_recon,&phase_encoding_coords).grid(&grid_size);
    cfl::from_array(work_dir.join(format!("rank")), &gridded.into_dyn()).unwrap();

    let var = cfl::to_array(work_dir.join(format!("var")),true).unwrap();
    let mut var_recon = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
    let v_slice = var_recon.as_slice_memory_order_mut().unwrap();
    var.as_slice_memory_order().unwrap()
    .iter()
    .zip(sample_ordering.iter())
    .for_each(|(elem,order)|{
        v_slice[order.linear_idx] = *elem;
    });
    let gridded = KSpace::from_array2(var_recon,&phase_encoding_coords).grid(&grid_size);
    cfl::from_array(work_dir.join(format!("var")), &gridded.into_dyn()).unwrap();

    let re = cfl::to_array(work_dir.join(format!("removed_energy")),true).unwrap();
    let mut re_recon = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
    let re_slice = re_recon.as_slice_memory_order_mut().unwrap();
    re.as_slice_memory_order().unwrap()
    .iter()
    .zip(sample_ordering.iter())
    .for_each(|(elem,order)|{
        re_slice[order.linear_idx] = *elem;
    });
    let gridded = KSpace::from_array2(re_recon,&phase_encoding_coords).grid(&grid_size);
    cfl::from_array(work_dir.join(format!("removed_energy")), &gridded.into_dyn()).unwrap();


}

//cargo test --release --package kspace-mppca --bin kmppca -- reduce_noise --exact --nocapture
#[test]
fn reduce_noise() {

    let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
    let n_vols = 67;
    let nx = 590;
    let chunk_size = 1000;
    let work_dir = kspace_dir.parent().unwrap().join("kmppca");

    let phase_encoding_coords = ViewTable::from_file(work_dir.join("views"))
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    let samples_per_vol = nx*phase_encoding_coords.len();
    
    let readers:Vec<_> = (0..n_vols).into_par_iter().map(|vol_idx|{
        CflReader::new(work_dir.join(format!("f{vol_idx}"))).unwrap()
    }).collect();

    let mut writers:Vec<_> = (0..n_vols).into_par_iter().map(|vol_idx|{
        CflWriter::new(work_dir.join(format!("fd{vol_idx}")),&[samples_per_vol]).unwrap()
    }).collect();

    let mut rank = CflWriter::new(work_dir.join("rank"),&[samples_per_vol]).unwrap();
    let mut rank_tmp = vec![Complex32::ZERO;chunk_size];

    let mut var = CflWriter::new(work_dir.join("var"),&[samples_per_vol]).unwrap();
    let mut var_tmp = vec![Complex32::ZERO;chunk_size];

    let mut removed_energy = CflWriter::new(work_dir.join("removed_energy"),&[samples_per_vol]).unwrap();
    let mut removed_energy_tmp = vec![Complex32::ZERO;chunk_size];


    let sample_indices:Vec<usize> = (0..samples_per_vol).collect();
    
    for (i,chunk) in sample_indices.chunks(chunk_size).enumerate() {
        println!("working on {} of {}",i,ceiling_div(samples_per_vol, chunk_size));
        let mut c_mat = extract_data(&chunk, &readers);
        //println!("shape: {:?}",c_mat.shape());
        //let _ = singular_value_threshold_mppca(&mut c_mat, None);

        let info = mp_denoise_matrix(&mut c_mat, None, None);
        //let info = mp_denoise_matrix(&mut c_mat, Some(1.6));

        rank_tmp.fill(Complex32::new(info.rank as f32,0.));
        rank.write_from(chunk,&rank_tmp).unwrap();

        var_tmp.fill(Complex32::new(info.variance.unwrap_or(0.) as f32,0.));
        var.write_from(chunk,&var_tmp).unwrap();

        removed_energy_tmp.fill(Complex32::new(info.init_energy - info.final_energy,0.));
        removed_energy.write_from(chunk,&removed_energy_tmp).unwrap();

        insert_data(&chunk, &mut writers, &c_mat);

    }

}











fn k_range(n:usize) -> Range<i32> {
    assert!(n!=0,"n must be greater than 0");
    if n%2 == 0 {
        -((n/2) as i32) .. ((n/2) as i32)
    }else {
        -((n/2) as i32) .. ceiling_div(n, 2) as i32
    }
}

pub fn ceiling_div(a:usize,b:usize) -> usize {
    (a + b - 1) / b
}

#[derive(Serialize,Deserialize)]
struct SampleOrdering {
    linear_idx:usize,
    // value of the norm that is used to rank the samples
    norm:i32,
}