use std::{collections::btree_map::RangeMut, fs::{create_dir_all, File}, io::{Read, Write}, ops::Range, path::{Path, PathBuf}, time::Instant};

use cfl::{ndarray::{parallel::prelude::*, Array1, Array2, Axis, ShapeBuilder}, num_complex::Complex32, CflReader, CflWriter};
use cs_table::ViewTable;
use mr_data::kspace::KSpace;
use ndarray_linalg::SVDDC;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

pub struct DenoiseInfo {
    pub singular_values:Vec<Complex32>,
    pub estimated_rank:usize,
    pub estimated_noise:Option<f32>,
    pub init_energy:f32,
    pub final_energy:f32,
}

pub fn singular_value_threshold_mppca(matrix:&mut Array2<Complex32>, rank:Option<usize>) -> DenoiseInfo {

    let m = matrix.shape()[0];
    let n = matrix.shape()[1];

    let init_energy = matrix.iter().map(|x|x.norm_sqr()).sum::<f32>();

    //let mut _s = Array2::from_elem((m,n), Complex32::ZERO);

    let nn = m.min(n);
    let mut _s = Array2::from_elem((nn,nn), Complex32::ZERO);

    //let (_,s,_) = matrix.svd(true, true).unwrap();
    // let (_,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
    // matrix.par_mapv_inplace(|x| x / sigma_sq);
    //let (u,mut s,v) = matrix.svd(true, true).unwrap();

    let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();

    let u = u.unwrap();
    let v = v.unwrap();

    // println!("u:{:?}",u.shape());
    // println!("s:{:?}",s.shape());
    // println!("v:{:?}",v.shape());

    let mut noise = None;

    let rank = rank.unwrap_or_else(||{
        let (rank,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
        noise = Some(sigma_sq);
        rank
    });
    
    let singular_values:Vec<_> = s.iter().map(|&s|Complex32::new(s, 0.)).collect();

    s.iter_mut().enumerate().for_each(|(i,val)| if i >= rank {*val = 0.});

    //let u = u.unwrap();
    //let v = v.unwrap();
    let mut diag_view = _s.diag_mut();
    diag_view.assign(
        &s.map(|x|Complex32::new(*x,0.))
    );

    // if u in 1x1
    let denoised_matrix = if u.len() == 1 {
        _s.dot(&v)
    }else {
        u.dot(&_s).dot(&v)
    };

    //let denoised_matrix = u.dot(&_s).dot(&v);
    matrix.assign(&denoised_matrix);

    let final_energy = matrix.iter().map(|x|x.norm_sqr()).sum::<f32>();

    //let energy_makeup = (init_energy / final_energy).sqrt();
    //matrix.iter_mut().for_each(|x| *x *= energy_makeup);

    DenoiseInfo {
        singular_values: singular_values,
        estimated_rank: rank,
        estimated_noise: noise,
        init_energy,
        final_energy,
    }
}

/// returns the index of the singular value to threshold, along with the estimated matrix variance
fn marchenko_pastur_singular_value(singular_values:&[f32],m:usize,n:usize) -> (usize,f32) {
    
    let r = m.min(n);
    let mut vals = singular_values.to_owned();

    let scaling:Vec<_> = (0..r).clone().map(|x| (m.max(n) as f32 - x as f32) / n as f32).collect();
    
    vals.iter_mut().for_each(|x| *x = x.powi(2) / n as f32);
    vals.reverse();
    let mut csum = cumsum(&vals);
    csum.reverse();
    vals.reverse();

    let cmean:Vec<_> = csum.iter().zip((1..r+1).rev()).map(|(x,y)| x / y as f32).collect();
    let sigmasq_1:Vec<_> = cmean.iter().zip(scaling.iter()).map(|(x,y)| *x / *y).collect();
    let range_mp:Vec<_> = (0..r).map(|x| x as f32).map(|x| (m as f32 - x) /  n as f32).map(|x| x.sqrt() * 4.).collect();
    let range_data:Vec<_> = vals[0..r].iter().map(|x| x - vals[r-1]).collect();
    let sigmasq_2:Vec<_> = range_data.into_iter().zip(range_mp).map(|(x,y)|x / y).collect();

    //println!("{:?}",sigmasq_1);
    //println!("{:?}",sigmasq_2);

    let idx = sigmasq_1.iter().zip(sigmasq_2).enumerate().find_map(|(i,(s1,s2))|{
        if s2 < *s1 {
            Some(i)
        }else {
            None
        }
    }).unwrap();

    let variance_estimate = sigmasq_1[idx];

    (idx,variance_estimate)
}

fn cumsum(x:&[f32]) -> Vec<f32> {
    let mut x = x.to_owned();
    let mut s = 0.;
    x.iter_mut().for_each(|val|{
        *val += s;
        s = *val;
    });
    x
}

pub fn extract_data(idx:&[usize],readers:&[CflReader]) -> Array2<Complex32> {
    let mut result = Array2::<Complex32>::zeros((idx.len(),readers.len()).f());
    result.axis_iter_mut(Axis(1)).into_par_iter().zip(readers.par_iter()).for_each(|(mut col,reader)|{
        reader.read_into(&idx, col.as_slice_memory_order_mut().unwrap()).unwrap();
    });
    result
}

pub fn insert_data(idx:&[usize], writers:&mut [CflWriter], matrix:&Array2<Complex32>) {
    writers.par_iter_mut().zip(matrix.axis_iter(Axis(1)).into_par_iter()).for_each(|(w,col)|{
        w.write_from(idx, col.as_slice_memory_order().unwrap()).unwrap();
    });
}

/// find the low-rank approximation of a casorati matrix based on known noise levels
/// singular values are sorted greatest to least
fn mp_find_rank(singular_values:&[f32],sigma_sqr:f32,m:usize,n:usize) -> usize {
    assert!(m >= n,"casorati matrix must have at least as many rows as columns");
    let mut rank:usize = n;
    let mut cumulative_sum = 0.;
    for (i,val) in singular_values.iter().rev().enumerate() {
        let scale = (m - n + i - 1) as f32 / n as f32;
        let lamb = val.powi(2) / n as f32;
        cumulative_sum += lamb;
        let cumulative_mean = cumulative_sum / (i + 1) as f32;
        let var = cumulative_mean / scale;
        if var < sigma_sqr {
            rank -= 1;
        }
    }
    rank
}

#[test]
fn test_find_rank() {
    let singular_values:Vec<f32> = vec![267.922179,77.677573,77.367026,76.901532,76.218652,75.661715,74.796348,74.245850,73.357911,73.255173,72.823380,72.246012,71.902821,71.266427,70.512379,70.128508,69.673704,69.044314,68.675660,68.103519,67.717366,67.562810,66.985499,66.324698,66.120651,65.819462,65.727307,65.209190,64.870015,64.603728,64.274472,63.772005,63.499366,62.813258,62.612546,62.395864,61.785515,61.286978,60.677350,60.473087,60.263632,60.187295,59.388814,59.090708,58.686399,58.106206,57.738611,57.491265,57.080259,56.797701,56.341278,55.811282,55.283017,55.040263,53.932982,53.506302,53.326709,52.902144,52.633008,51.825234,51.082119,50.702511,50.321345,49.923459,49.815425,48.324251,47.976007];
    //let rank = find_rank(&singular_values, 4., 1000, 67);
    let (rank,noise) = mp_estimate_rank(&singular_values, 1000, 67);
    println!("rank = {}",rank);
    println!("noise = {}",noise);
}

// estimates the rank and variance simultaneousely
fn mp_estimate_rank(singular_values:&[f32],m:usize,n:usize) -> (usize,f32) {
    assert!(m >= n,"casorati matrix must have at least as many rows as columns");
    let mut rank:usize = n;
    let mut cumulative_sum = 0.;
    let mut noise_estimate = 0.;
    for (i,val) in singular_values.iter().rev().enumerate() {
        let scale = (m - n + i - 1) as f32 / n as f32;
        let lamb = val.powi(2) / n as f32;
        let lamb_last = singular_values.last().unwrap().powi(2) / n as f32;
        cumulative_sum += lamb;
        let cumulative_mean = cumulative_sum / (i + 1) as f32;
        let var = cumulative_mean / scale;
        let gamma = (m - ((n-1) - i)) as f32 / n as f32;
        let range_mp = 4. * gamma.sqrt();
        let range_data = lamb - lamb_last;
        let sig_estimate = range_data / range_mp;
        if sig_estimate < var {
            rank -= 1;
            noise_estimate = var;
        }
    }
    (rank,noise_estimate)
}



pub struct PCAInfo {
    pub init_energy:f32,
    pub final_energy:f32,
    pub rank:usize,
    pub variance:Option<f32>
}

fn denoise_matrix(matrix:&mut Array2<Complex32>,singular_value_threshold:f32) -> PCAInfo {

    // m is the number of features (rows)
    //let m = matrix.shape()[0];
    // n is the number observations (columns)
    let n = matrix.shape()[1];

    // get the initial energy of the matrix prior to noise reduction
    let init_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // calculate SVD
    let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();
    let u = u.unwrap();
    let v = v.unwrap();

    // retrieve mutable reference to the singular values
    let singular_values = s.as_slice_mut().unwrap();

    // returns None if the matrix has full rank
    let rank = singular_values.iter().enumerate()
    .find_map(|(rank,&lambda)| if lambda < singular_value_threshold {
        Some(rank)
    } else {
        None
    });

    if let Some(rank) = rank {
        singular_values[rank..].fill(0.);
    }
    // do hard thresholding by nullifying singular values above estimated rank
    
    // reconstruct full diagonal matrix from nullified singular values
    let s = Array2::from_diag(&s.map(|&x|Complex32::new(x, 0.)));

    // reconstruct data matrix and assign values to original
    let recon = u.dot(&s).dot(&v);
    matrix.assign(&recon);

    // find the resulting energy of the denoised matrix
    let final_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // record meta data and return
    PCAInfo {
        init_energy,
        final_energy,
        rank:rank.unwrap_or(n),
        variance: None,
    }

}


pub fn mp_denoise_matrix(matrix:&mut Array2<Complex32>,variance:Option<f32>,rank:Option<usize>) -> PCAInfo {

    // m is the number of features (rows)
    let m = matrix.shape()[0];
    // n is the number observations (columns)
    let n = matrix.shape()[1];

    // get the initial energy of the matrix prior to noise reduction
    let init_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // calculate SVD
    let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();
    let u = u.unwrap();
    let v = v.unwrap();

    // retrieve mutable reference to the singular values
    let singular_values = s.as_slice_mut().unwrap();


    // determine the rank and noise (if applicable) of the matrix
    let (rank,var) = match &variance {
        Some(var) => {
            let rank = if let Some(rank) = rank {
                rank
            }else {
                mp_find_rank(singular_values, *var, m, n)
            };
            (rank,*var)
        }
        None => {
            let (estimated_rank,var_estimate) = mp_estimate_rank(singular_values, m, n);
            (rank.unwrap_or(estimated_rank),var_estimate)
        }
    };




    // do hard thresholding by nullifying singular values above estimated rank
    singular_values[rank..].fill(0.);

    // reconstruct full diagonal matrix from nullified singular values
    let s = Array2::from_diag(&s.map(|&x|Complex32::new(x, 0.)));

    // reconstruct data matrix and assign values to original
    let recon = u.dot(&s).dot(&v);
    matrix.assign(&recon);

    // find the resulting energy of the denoised matrix
    let final_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // record meta data and return
    PCAInfo {
        init_energy,
        final_energy,
        rank,
        variance: Some(var),
    }

}

/// performs a monte-carlo simulation to find the largest singular value of a measured noise
/// matrix
fn get_largest_singular_value(casorati_matrix:&Array2<Complex32>) -> f32 {
    let (_,s,_) = casorati_matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();
    *s.first().unwrap()
}

fn estimate_largest_singular_value(noise_variance:f32,m:usize,n:usize,monte_carlo_n_iter:usize) -> f32 {

    // Create a normal distribution with the specified mean and standard deviation
    let normal_dist = Normal::new(0., noise_variance.sqrt()).expect("Failed to create normal distribution");

    // Create a random number generator
    let mut rng = thread_rng();

    // Create function that generates random complex-valued gaussian noise
    let mut sample_complex = || {
        Complex32::new(normal_dist.sample(&mut rng),normal_dist.sample(&mut rng))
    };

    let mut max_singular_value = 0.;
    for _ in 0..monte_carlo_n_iter {
        let complex_rand_matrix = Array2::<Complex32>::from_shape_simple_fn((m,n).f(), &mut sample_complex);
        let (_,s,_) = complex_rand_matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();
        let this_singular_value = *s.get(0).unwrap();
        if this_singular_value > max_singular_value {
            max_singular_value = this_singular_value;
        }
    }

    max_singular_value

}

#[test]
fn noise_matrix() {
    let now = Instant::now();
    let max_val = estimate_largest_singular_value(0.6752, 1000, 67, 10);
    let later = now.elapsed();
    println!("took {} ms",later.as_millis());
    println!("max val = {}",max_val);
}



pub struct KSpacePrepInputs {
    pub kspace_dir: PathBuf,
    pub work_dir: PathBuf,
    pub n_volumes:usize,
}

pub struct KSpacePrepOutputs {
    pub work_dir:PathBuf,
    pub phase_encoding_table_path:PathBuf,
    pub n_volumes:usize,
    pub nx:usize,
}

//cargo test --release --package kspace-mppca --bin kmppca -- resolve_input --exact --nocapture 
pub fn prepare_input_from_kspace_dir(inputs:KSpacePrepInputs) -> KSpacePrepOutputs {

    // defined by this process
    let phase_encoding_table_name = "views";

    create_dir_all(&inputs.work_dir).expect("failed to create work dir");
    let kspace_files:Vec<_> = (0..inputs.n_volumes).map(|i|inputs.kspace_dir.join(format!("k{}",i)).join("k0")).collect();
    // load first kspace to get the common phase encoding coordinates
    let k = KSpace::from_file(&kspace_files[0]).expect("failed to load kspace data");
    let phase_encoding_coords_raw = k.coords();
    let nx = k.line_len();
    kspace_files.par_iter().for_each(|file|{
        let ksp = KSpace::from_file(file).expect("failed to load raw kspace data");
        let a = ksp.to_array2(&phase_encoding_coords_raw).into_dyn();
        let compressed_kspace_file = inputs.work_dir.join(file.parent().unwrap().file_name().unwrap());
        cfl::from_array(compressed_kspace_file, &a).unwrap();
    });

    let pe_table_path = inputs.work_dir.join(phase_encoding_table_name);
    ViewTable::from_coord_pairs(&phase_encoding_coords_raw).unwrap().write(&pe_table_path).unwrap();

    KSpacePrepOutputs {
        work_dir: inputs.work_dir,
        phase_encoding_table_path: pe_table_path,
        n_volumes: inputs.n_volumes,
        nx,
    }

}

pub fn prepare_from_kspace_files(files:&[PathBuf],work_dir:impl AsRef<Path>) -> KSpacePrepOutputs {

    let work_dir = work_dir.as_ref();
    let phase_encoding_table_name = "views";

    let n_volumes = files.len();

    create_dir_all(&work_dir).expect("failed to create work dir");
    // load first kspace to get the common phase encoding coordinates
    let k = KSpace::from_file(&files[0]).expect("failed to load kspace data");
    let phase_encoding_coords_raw = k.coords();
    let nx = k.line_len();
    files.par_iter().for_each(|file|{
        let ksp = KSpace::from_file(file).expect("failed to load raw kspace data");
        let a = ksp.to_array2(&phase_encoding_coords_raw).into_dyn();
        let compressed_kspace_file = work_dir.join(file.parent().unwrap().file_name().unwrap());
        cfl::from_array(compressed_kspace_file, &a).unwrap();
    });

    let pe_table_path = work_dir.join(phase_encoding_table_name);
    ViewTable::from_coord_pairs(&phase_encoding_coords_raw).unwrap().write(&pe_table_path).unwrap();

    KSpacePrepOutputs {
        work_dir: work_dir.to_path_buf(),
        phase_encoding_table_path: pe_table_path,
        n_volumes,
        nx,
    }

}



// //cargo test --release --package kspace-mppca --bin kmppca -- parse_headfiles --exact --nocapture 
// #[test]
// fn parse_headfiles() {
//     let kspace_dir = Path::new("/Users/Wyatt/scratch/S69964/object-data");
//     let n_vols = 67;

//     let headfiles:Vec<_> = (0..n_vols).map(|i|kspace_dir.join(format!("{}.headfile",i))).collect();

//     let bval_key = "bvalue";
//     let b_values:Vec<_> = headfiles.par_iter().map(|file|{
//         let hf = Headfile::open(file).to_hash();
//         let bval = hf.get(bval_key)
//         .expect(&format!("failed to get {} from headfile",bval_key));
//         bval.parse::<f32>().expect("failed to parse b-value to float")
//     }).collect();

//     println!("bvals: {:#?}",b_values);

//     let max_bval = *b_values.iter().max_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();
//     let min_bval = *b_values.iter().min_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();

//     let b0_indices = [0,11,22,33,44,55];

// }


#[derive(Debug,Clone,Copy)]
pub enum SampleOrderingNorm {
    // l1 (manhattan) norm
    L1,
    // l2 squared (squared euclidian norm)
    L2,
    /// infinity norm (chebychev norm)
    LInf,
}



pub struct FlattenSamplesInputs {
    pub sample_ordering_norm:SampleOrderingNorm,
    pub file_prefix:String,
    pub work_dir:PathBuf,
    pub n_volumes:usize,
    pub nx:usize,
    pub phase_encoding_table_path:PathBuf,
}

pub struct FlattenSamplesOutputs {
    pub sample_ordering_file:PathBuf,
    pub file_prefix:String,
    pub work_dir:PathBuf,
    pub n_volumes:usize,
    pub nx:usize,
}

//cargo test --release --package kspace-mppca --bin kmppca -- flatten --exact --nocapture

pub fn flatten(inputs:FlattenSamplesInputs) -> FlattenSamplesOutputs {

    let nx = inputs.nx;
    let n_vols = inputs.n_volumes;
    let pe_table_path = &inputs.phase_encoding_table_path;
    let work_dir = &inputs.work_dir;
    let file_prefix = &inputs.file_prefix;

    // defined by this process
    let sample_ordering_file = "sample_ordering";

    let phase_encoding_coords = ViewTable::from_file(pe_table_path)
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    let norm = match inputs.sample_ordering_norm {
        SampleOrderingNorm::L1 => |kx:i32,ky:i32,kz:i32| -> i32 {
            kx.abs() + ky.abs() + kz.abs()
        },
        SampleOrderingNorm::L2 => |kx:i32,ky:i32,kz:i32| -> i32 {
            kx*kx + ky*ky + kz*kz
        },
        SampleOrderingNorm::LInf => |kx:i32,ky:i32,kz:i32| -> i32 {
            kx.abs().max(ky.abs()).max(kz.abs())
        },
    };

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

    let ordering_file_path = work_dir.join(sample_ordering_file);

    let mut ordering_fle = File::create(&ordering_file_path).unwrap();
    ordering_fle.write_all(
        &bincode::serialize(&sample_ordering).unwrap()
    ).unwrap();

    // sort k-space samples by their coordinate norm and write to 1-D cfl file
    (0..n_vols).into_par_iter().for_each(|vol_idx|{
        println!("flattening volume {}",vol_idx);
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
        cfl::from_array(work_dir.join(format!("{file_prefix}{vol_idx}")), &sorted.into_dyn()).unwrap();
    });

    FlattenSamplesOutputs {
        sample_ordering_file: ordering_file_path,
        file_prefix:inputs.file_prefix,
        work_dir: inputs.work_dir,
        n_volumes: n_vols,
        nx,
    }

}



pub fn estimate_variance(inputs:&DenoiseInputs) -> f32 {

    let chunk_size = inputs.sample_chunk_size;
    let assumed_variance:Option<f32> = inputs.assumed_variance;
    let assumed_rank:Option<usize> = inputs.assumed_global_rank;

    // dependencies
    let n_vols = inputs.n_volumes;
    let work_dir = &inputs.work_dir;
    let flattened_file_prefix = &inputs.flattened_file_prefix;

    let dims = cfl::get_dims(work_dir.join(format!("{flattened_file_prefix}{}",0))).unwrap();
    let samples_per_vol = dims.iter().product();

    let readers:Vec<_> = (0..n_vols).into_par_iter().map(|vol_idx|{
        CflReader::new(work_dir.join(format!("{flattened_file_prefix}{vol_idx}"))).unwrap()
    }).collect();

    // sample indices that are segmented and processed by chunks
    let sample_indices:Vec<usize> = (0..samples_per_vol).collect();
    
    let mut n_rank_0 = 0;
    let mut max_variance = 0.;

    for (i,chunk) in sample_indices.chunks(chunk_size).enumerate() {
        println!("working on {} of {}",i+1,ceiling_div(samples_per_vol, chunk_size));

        // extract neighborhood from sample readers
        let mut c_mat = extract_data(&chunk, &readers);

        // reduce noise in casorati matrix via hard-thresholding singular values
        // return an info struct that contains meta data on what was performed
        let info = mp_denoise_matrix(&mut c_mat, assumed_variance, assumed_rank);

        // find the max variance over all rank 0 detections
        if info.rank == 0 {
            let v = info.variance.unwrap_or(0.);
            if v > max_variance {
                max_variance = v;
            }
            n_rank_0 += 1;
        }

    }
    println!("n rank 0 neighborhoods: {}",n_rank_0);
    println!("estimated noise variane: {}",max_variance);
    max_variance
}



pub struct UnflattenInputs {
    pub grid_size:[usize;3],
    pub sample_ordering_file:PathBuf,
    pub work_dir:PathBuf,
    pub n_volumes:usize,
    pub nx:usize,
    pub phase_encoding_table_path:PathBuf,
    pub denoised_file_prefix:String,
    pub rank_filepath:PathBuf,
    pub variance_filepath:PathBuf,
    pub removed_energy_filepath:PathBuf,
    pub neighborhood_filepath:PathBuf,
}


//cargo test --release --package kspace-mppca --bin kmppca -- unflatten --exact --nocapture

pub fn unflatten(inputs:UnflattenInputs) {

    let phase_encoding_filepath = &inputs.phase_encoding_table_path;
    let work_dir = &inputs.work_dir;
    let sample_ordering_file = &inputs.sample_ordering_file;
    let n_vols = inputs.n_volumes;
    let denoised_prefix = &inputs.denoised_file_prefix;
    let nx = inputs.nx;
    let rank_filepath = &inputs.rank_filepath;
    let variance_filepath = &inputs.variance_filepath;
    let removed_energy_filepath = &inputs.removed_energy_filepath;
    let neighborhood_filepath = &inputs.neighborhood_filepath;
    let grid_size = &inputs.grid_size;

    let result_prefix = "o";

    let phase_encoding_coords = ViewTable::from_file(phase_encoding_filepath)
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    let mut ordering_fle = File::open(sample_ordering_file).unwrap();
    let mut bytes:Vec<u8> = vec![];
    ordering_fle.read_to_end(
        &mut bytes
    ).unwrap();
    let sample_ordering:Vec<SampleOrdering> = bincode::deserialize(&bytes).unwrap();


    // sort k-space samples by their coordinate norm and write to 1-D cfl file
    (0..n_vols).into_par_iter().for_each(|vol_idx|{
        println!("reconstructing volume {}",vol_idx);
        let denoised_filename = work_dir.join(format!("{denoised_prefix}{vol_idx}"));
        let result_filename = work_dir.join(format!("{result_prefix}{vol_idx}"));
        unflatten_grid(&denoised_filename,result_filename,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    });

    unflatten_grid(rank_filepath,rank_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(variance_filepath,variance_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(removed_energy_filepath,removed_energy_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(neighborhood_filepath,neighborhood_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);

}

pub fn unflatten_to_kspace(inputs:UnflattenInputs,kspace_files:&[PathBuf]) {

    let phase_encoding_filepath = &inputs.phase_encoding_table_path;
    let work_dir = &inputs.work_dir;
    let sample_ordering_file = &inputs.sample_ordering_file;
    let n_vols = inputs.n_volumes;
    let denoised_prefix = &inputs.denoised_file_prefix;
    let nx = inputs.nx;
    let rank_filepath = &inputs.rank_filepath;
    let variance_filepath = &inputs.variance_filepath;
    let removed_energy_filepath = &inputs.removed_energy_filepath;
    let neighborhood_filepath = &inputs.neighborhood_filepath;
    let grid_size = &inputs.grid_size;

    let phase_encoding_coords = ViewTable::from_file(phase_encoding_filepath)
    .unwrap()
    .coordinate_pairs::<i32>()
    .unwrap();

    let mut ordering_fle = File::open(sample_ordering_file).unwrap();
    let mut bytes:Vec<u8> = vec![];
    ordering_fle.read_to_end(
        &mut bytes
    ).unwrap();
    let sample_ordering:Vec<SampleOrdering> = bincode::deserialize(&bytes).unwrap();

    assert_eq!(n_vols,kspace_files.len(),"mismatch between number of volumes and specified outputs");
    // sort k-space samples by their coordinate norm and write to 1-D cfl file
    (0..n_vols).into_par_iter().zip(kspace_files.par_iter()).for_each(|(vol_idx,kspace_file)|{
        println!("reconstructing volume {}",vol_idx);
        let denoised_filename = work_dir.join(format!("{denoised_prefix}{vol_idx}"));
        unflatten_to_ksp(&denoised_filename,kspace_file,nx,&phase_encoding_coords,&sample_ordering);
    });

    unflatten_grid(rank_filepath,rank_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(variance_filepath,variance_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(removed_energy_filepath,removed_energy_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
    unflatten_grid(neighborhood_filepath,neighborhood_filepath,nx,&phase_encoding_coords,&sample_ordering,grid_size);
}

/// unflattens a 1-D cfl into a grid
fn unflatten_grid(file_name:impl AsRef<Path>,output_filename:impl AsRef<Path>,nx:usize,phase_encoding_coords:&[[i32;2]],sample_ordering:&[SampleOrdering],grid_size:&[usize]) {
    let rank = cfl::to_array(&file_name,true).unwrap();
    let mut recon_2d = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
    let r_slice = recon_2d.as_slice_memory_order_mut().unwrap();
    rank.as_slice_memory_order().unwrap()
    .iter()
    .zip(sample_ordering.iter())
    .for_each(|(elem,order)|{
        r_slice[order.linear_idx] = *elem;
    });
    let gridded = KSpace::from_array2(recon_2d,&phase_encoding_coords).grid(&grid_size);
    cfl::from_array(output_filename, &gridded.into_dyn()).unwrap();
}

fn unflatten_to_ksp(file_name:impl AsRef<Path>,output_filename:impl AsRef<Path>,nx:usize,phase_encoding_coords:&[[i32;2]],sample_ordering:&[SampleOrdering]) {
    let rank = cfl::to_array(&file_name,true).unwrap();
    let mut recon_2d = Array2::<Complex32>::from_elem((nx,phase_encoding_coords.len()).f(), Complex32::ZERO);
    let r_slice = recon_2d.as_slice_memory_order_mut().unwrap();
    rank.as_slice_memory_order().unwrap()
    .iter()
    .zip(sample_ordering.iter())
    .for_each(|(elem,order)|{
        r_slice[order.linear_idx] = *elem;
    });
    let ksp = KSpace::from_array2(recon_2d,&phase_encoding_coords);
    ksp.write_to_file(output_filename).unwrap();
}

pub struct DenoiseInputs {
    pub sample_chunk_size:usize,
    pub assumed_variance:Option<f32>,
    pub assumed_global_rank:Option<usize>,
    pub flattened_file_prefix:String,
    pub work_dir:PathBuf,
    pub n_volumes:usize,
}

pub struct DenoiseOutputs {
    pub work_dir:PathBuf,
    pub n_volumes:usize,
    pub denoised_file_prefix:String,
    pub rank_filepath:PathBuf,
    pub variance_filepath:PathBuf,
    pub removed_energy_filepath:PathBuf,
    pub neighborhood_filepath:PathBuf,
}

//cargo test --release --package kspace-mppca --bin kmppca -- reduce_noise --exact --nocapture
pub fn reduce_noise(inputs:DenoiseInputs) -> DenoiseOutputs {

    // defined by process
    let chunk_size = inputs.sample_chunk_size;
    let assumed_variance:Option<f32> = inputs.assumed_variance;
    let assumed_rank:Option<usize> = inputs.assumed_global_rank;

    let denoised_file_prefix = "fd";
    let rank_filename = "rank";
    let variance_filename = "variance";
    let removed_energy_filename = "removed_energy";
    let neighborhood_filename = "low_rank_neighborhood";

    // dependencies
    let n_vols = inputs.n_volumes;
    let work_dir = &inputs.work_dir;
    let flattened_file_prefix = &inputs.flattened_file_prefix;

    let dims = cfl::get_dims(work_dir.join(format!("{flattened_file_prefix}{}",0))).unwrap();
    let samples_per_vol = dims.iter().product();

    let readers:Vec<_> = (0..n_vols).into_par_iter().map(|vol_idx|{
        CflReader::new(work_dir.join(format!("{flattened_file_prefix}{vol_idx}"))).unwrap()
    }).collect();

    let mut writers:Vec<_> = (0..n_vols).into_par_iter().map(|vol_idx|{
        CflWriter::new(work_dir.join(format!("{denoised_file_prefix}{vol_idx}")),&[samples_per_vol]).unwrap()
    }).collect();

    let rank_filepath = work_dir.join(rank_filename);
    let mut rank = CflWriter::new(&rank_filepath,&[samples_per_vol]).unwrap();
    let mut rank_tmp = vec![Complex32::ZERO;chunk_size];

    let variance_filepath = work_dir.join(variance_filename);
    let mut var = CflWriter::new(&variance_filepath,&[samples_per_vol]).unwrap();
    let mut var_tmp = vec![Complex32::ZERO;chunk_size];

    let removed_energy_filepath = work_dir.join(removed_energy_filename);
    let mut removed_energy = CflWriter::new(&removed_energy_filepath,&[samples_per_vol]).unwrap();
    let mut removed_energy_tmp = vec![Complex32::ZERO;chunk_size];

    let neighborhood_filepath = work_dir.join(neighborhood_filename);
    let mut low_rank_neighborhood = CflWriter::new(&neighborhood_filepath,&[samples_per_vol]).unwrap();
    let mut low_rank_neighborhood_tmp = vec![Complex32::ZERO;chunk_size];
    
    // sample indices that are segmented and processed by chunks
    let sample_indices:Vec<usize> = (0..samples_per_vol).collect();
    
    for (i,chunk) in sample_indices.chunks(chunk_size).enumerate() {
        println!("working on {} of {}",i+1,ceiling_div(samples_per_vol, chunk_size));

        // extract neighborhood from sample readers
        let mut c_mat = extract_data(&chunk, &readers);

        // reduce noise in casorati matrix via hard-thresholding singular values
        // return an info struct that contains meta data on what was performed
        let info = mp_denoise_matrix(&mut c_mat, assumed_variance, assumed_rank);

        // write the measured (or assumed low-rank) approximation
        rank_tmp.fill(Complex32::new(info.rank as f32,0.));
        rank.write_from(chunk,&rank_tmp).unwrap();

        // write the measured (or assumed) variance
        var_tmp.fill(Complex32::new(info.variance.unwrap_or(0.) as f32,0.));
        var.write_from(chunk,&var_tmp).unwrap();

        // write the removed energy
        removed_energy_tmp.fill(Complex32::new(info.init_energy - info.final_energy,0.));
        removed_energy.write_from(chunk,&removed_energy_tmp).unwrap();

        // write the neighborhood index
        low_rank_neighborhood_tmp.fill(Complex32::new(i as f32,0.));
        low_rank_neighborhood.write_from(chunk,&low_rank_neighborhood_tmp).unwrap();

        insert_data(&chunk, &mut writers, &c_mat);

    }

    DenoiseOutputs {
        work_dir: inputs.work_dir,
        n_volumes: n_vols,
        denoised_file_prefix: denoised_file_prefix.to_string(),
        rank_filepath,
        variance_filepath,
        removed_energy_filepath,
        neighborhood_filepath,
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
