#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = A pi(t) + (1 / (2N))
 *
 */
__global__ void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    // TODO: fill in the kernel code here
    const unsigned int i = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < num_nodes) {
        float sum = 0;
	for (unsigned int j = graph_indices[i]; j < graph_indices[i+1]; j++) {
            sum += graph_nodes_in[graph_edges[j]] * inv_edges_per_node[graph_edges[j]];
	}
        graph_nodes_out[i] = 0.5 / (float) num_nodes + 0.5 * sum;
    }
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    // TODO: allocate GPU memory
    float *device_input_array = nullptr;
    float *device_output_array = nullptr;
    float *device_invs = nullptr;
    float *temp = new float[num_nodes];
    uint *device_edges = nullptr;
    uint *device_indices = nullptr;
    const size_t num_bytes = num_nodes * sizeof(float);
    // TODO: check for allocation failure
    cudaMalloc(&device_input_array, num_bytes);
    cudaMalloc(&device_output_array, num_bytes);
    cudaMalloc(&device_invs, num_bytes);
    cudaMalloc(&device_edges, num_nodes * avg_edges * sizeof(uint));
    cudaMalloc(&device_indices, (num_nodes+1)*sizeof(uint));
    if (!device_input_array || !device_output_array || !device_invs || !device_edges || !device_indices) {
        std::cerr << "Error in allocating GPU memory.\n";
        return 1;
    }

    // TODO: copy data to the GPU
    cudaMemcpy(device_input_array, h_node_values_input, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_invs, h_inv_edges_per_node, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_edges, h_graph_edges, num_nodes * avg_edges * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, h_graph_indices, (num_nodes + 1) * sizeof (uint), cudaMemcpyHostToDevice);
    check_launch("copy to gpu");


    // launch kernels
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;

    // TODO: launch your kernels the appropriate number of iterations
    int numBlocks = (num_nodes + block_size - 1) / block_size;

    for (uint i = 0; i < nr_iterations / 2; i++) {
        device_graph_propagate<<<numBlocks, block_size>>>(device_indices, device_edges, device_input_array, device_output_array, device_invs, num_nodes);
	device_graph_propagate<<<numBlocks, block_size>>>(device_indices, device_edges, device_output_array, device_input_array, device_invs, num_nodes);
    }
    if (nr_iterations % 2 == 1) {
        device_graph_propagate<<<numBlocks, block_size>>>(device_indices, device_edges, device_input_array, device_output_array, device_invs, num_nodes);
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    if (nr_iterations % 2 == 1) {
        cudaMemcpy(h_gpu_node_values_output, device_output_array, num_nodes*sizeof(float), cudaMemcpyDeviceToHost);
    } else {
	cudaMemcpy(h_gpu_node_values_output, device_input_array, num_nodes*sizeof(float), cudaMemcpyDeviceToHost);
    }
    check_launch("copy from gpu");
    // TODO: free the memory you allocated!
    cudaFree(device_input_array);
    cudaFree(device_output_array);
    cudaFree(device_edges);
    cudaFree(device_invs);
    cudaFree(device_indices);

    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    // TODO
    uint total = (edges*2-1+3) * sizeof(float) + (edges+1) * sizeof(uint);
    return total * nodes * iterations;
}

#endif
