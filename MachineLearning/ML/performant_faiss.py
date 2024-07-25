import faiss  
import numpy as np  
faiss.omp_set_num_threads(20)  
# Load vectors from the .npy file  
vectors_2d = np.load("vectors_2d.npy")
d = vectors_2d.shape[1]  # Dimension of the vectors  
factory_string = "IVF65536,PQ8x8"  # not bad original
index_fpq = faiss.index_factory(d, factory_string, faiss.METRIC_INNER_PRODUCT)  

# Move the index to GPU  
res = faiss.StandardGpuResources()  
gpu_index_fpq = faiss.index_cpu_to_gpu(res, 0, index_fpq)  
# Train the index on GPU  
gpu_index_fpq.train(vectors_2d)  
# Add vectors to the index  
gpu_index_fpq.add(vectors_2d)  

# Move the index back to CPU  
cpu_index_fpq = faiss.index_gpu_to_cpu(gpu_index_fpq)  
# write
faiss.write_index(cpu_index_fpq, "index_fpq_cpu.index")  
# read
cpu_index_fpq = faiss.read_index("index_fpq_cpu.index")