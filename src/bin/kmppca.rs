use core::ops::Range;
use std::path::Path;
use cfl::{CflReader, CflWriter};
use cfl::{ndarray::ShapeBuilder, num_complex::Complex32};
use cfl::ndarray::{Array1, Array2};
use kspace_mppca::{extract_data, insert_data, singular_value_threshold_mppca};
use mr_data::kspace::KSpace;
use rand_distr::ChiSquaredError;
use recon2::{config::config::TomlConfig, object_manager::object_manager::ObjectManager};

fn main() {

    let work_dir = "/privateShares/wa41/denoise_test2";

    let object_manager = ObjectManager::from_file("/privateShares/wa41/24.chdi.01.work/240725-12-1/S69963/object-data/object-manager").unwrap();
    let total_objs = object_manager.total_objects().unwrap();

    println!("total objects: {}",total_objs);

    // extract view coordinates
    let k = object_manager.kspace(0).expect("failed to get kspace 0");

    if k.len() > 0 {
        println!("warning! Only using first kspace object!");
    }

    let phase_encoding_coords = k[0].coords();
    let n_views = phase_encoding_coords.len();
    let nx = k[0].line_len();
    println!("nx: {}",nx);
    println!("n views: {}",n_views);

    let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
        kx*kx + ky*ky + kz*kz
    };

    // let norm = |kx:i32,ky:i32,kz:i32| -> i32 {
    //     kx.abs().max(ky.abs()).max(kz.abs())
    // };

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
    let n_samples_per_vol = idx;
    println!("n_samples_per_vol: {}",n_samples_per_vol);
    let chunk_size = 1000;

    sample_ordering.sort_by_key(|order|order.norm);
    
    let mut flattened = vec![];
    let mut denoised = vec![];
    let mut outputs = vec![];


    // encode
    for i in 1..11 {
        println!("{}",i);
        let ksp = object_manager.kspace(i).unwrap().remove(0);
        //let ksp = KSpace::from_file(format!("k{:02}",i)).unwrap();
        let c = ksp.to_array2(&phase_encoding_coords).into_dyn();
        let mut sorted = Array1::<Complex32>::from_elem(c.len().f(), Complex32::ZERO);
        let c_slice = c.as_slice_memory_order().unwrap();
        sorted.iter_mut().zip(sample_ordering.iter()).for_each(|(elem,order)|{
            //println!("r.0 = {}",r.0);
            *elem = c_slice[order.linear_idx]
        });

        let p = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"r"));
        let d = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"d"));
        let o = Path::new(work_dir).join(civm_rust_utils::num_label(i,total_objs,"k"));

        cfl::from_array(
            &p,
            &sorted.into_dyn()
        ).unwrap();

        flattened.push(p);
        denoised.push(d);
        outputs.push(o);

    }

    // denoise
    // let skip_indices = [0, 11, 22, 33, 44, 55]; // Indices to skip

    // let filtered_range: Vec<usize> = (0..67)
    //     .filter(|x| !skip_indices.contains(x)) // Filter out the unwanted indices
    //     .collect();

    let readers:Vec<_> = flattened.iter().map(|file|{
        CflReader::new(file).unwrap()
    }).collect();

    let mut writers:Vec<_> = denoised.iter().map(|file|{
        CflWriter::new(file,&[n_samples_per_vol]).unwrap()
    }).collect();
 
    //let mut init_energy = CflWriter::new("init_energy",&[16200]).unwrap();
    //let mut final_energy = CflWriter::new("final_energy",&[16200]).unwrap();

    let n_chunks = ceiling_div(16200*590, chunk_size);

    let mut singular_values = CflWriter::new("singular_values",&[n_chunks]).unwrap();

    //let mut idx:Vec<usize> = (0..590).collect();
    //let mut sv_idx:Vec<usize> = (0..61).collect();
 
    let indices:Vec<usize> = (0..n_samples_per_vol).collect();
    
    for (i,chunk) in indices.chunks(chunk_size).enumerate() {
        println!("working on {}",i);
        let mut c_mat = extract_data(&chunk, &readers);
        //println!("shape: {:?}",c_mat.shape());
        //let _ = singular_value_threshold_mppca(&mut c_mat, None);
        let info = singular_value_threshold_mppca(&mut c_mat, None);
        let s_val = info.singular_values.get(0).unwrap();

        singular_values.write(i, *s_val).unwrap();

        insert_data(&chunk, &mut writers, &c_mat);
        //idx.iter_mut().for_each(|x| *x += 590);
        //sv_idx.iter_mut().for_each(|x| *x += 61);
    }
 
    // println!("flushing writers ...");
    // for w in writers {
    //     w.flush().expect("failed to flush writer");
    // }

    for (output,den) in outputs.iter().zip(denoised.iter()) {
        println!("reconstructing {}",den.display());
        let denoised = cfl::to_array(den, true).unwrap();
        let mut compressed = Array2::from_elem((nx,n_views).f(), Complex32::ZERO);
        let c = compressed.as_slice_memory_order_mut().unwrap();
        denoised.iter().zip(sample_ordering.iter()).for_each(|(x,r)|{
            c[r.linear_idx] = *x;
        });
        let ksp = KSpace::from_array2(compressed, &phase_encoding_coords);
        ksp.write_to_file(output).unwrap();
    }

    // decode

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


struct SampleOrdering {
    linear_idx:usize,
    // value of the norm that is used to rank the samples
    norm:i32,
}