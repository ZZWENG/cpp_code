#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"
#include "mpi.h"
#include "utils/common.h"


#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

real norms(NeuralNetwork& nn) {
  real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
  arma::Mat<real> A, B, C, D;

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0) {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<real> da1 = nn.W[1].t() * diff;

  arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg) {
  int N = yc.n_cols;
  real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  real data_loss = ce_sum / N;
  real reg_loss = 0.5 * reg * norms(nn);
  real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads) {
  real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h;
        feedforward(nn, X, numcache);
        real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward(nn, X, numcache);
        real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j) {
      real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        write_cpudata_tofile(nn, iter);
      }

      iter++;
    }
  }
}


void GPUfeedforward(NeuralNetwork &nn, real* d_X, int X_n_rows, int X_n_cols,
    real* d_W0, real* d_W1, real* d_b0, real* d_b1, real* d_a1, real* d_yc, real* d_z1, real* d_z2, real* dexp)
{
    int N = X_n_cols;
    GPUrepmat(d_b0, d_z1, nn.b[0].n_rows, N);
    real alpha = 1.0;
    real beta = 1.0;
    myGEMM(d_W0, d_X, d_z1, &alpha, &beta, nn.W[0].n_rows, N, X_n_rows);
    GPUsigmoid(d_z1, d_a1, nn.W[0].n_rows, N);
    GPUrepmat(d_b1, d_z2, nn.b[1].n_rows, N);
    myGEMM(d_W1, d_a1, d_z2, &alpha, &beta, nn.W[1].n_rows, N, nn.W[0].n_rows);

    arma::Mat<real> a2(nn.b[1].n_rows, N);
    GPUexp(d_z2, d_yc, a2.n_rows, a2.n_cols);
    GPUcolSum(d_yc, dexp, a2.n_rows, a2.n_cols);
    GPUdiv(dexp, d_yc, a2.n_rows, a2.n_cols);
}

void GPUbackprop(NeuralNetwork &nn, real* d_y, real* d_diff, real* d_yc, int y_n_rows, int y_n_cols, real reg,
                real* d_X, real* d_XT, int X_n_rows, int X_n_cols, real* d_a1, real* d_W0, real* d_W1, real* d_W1_t,
                real* d_tmp_db1, real* d_da1, real *d_tmp1, real *d_tmp2, real *d_dz1, real *d_tmp_db0, real *d_a1_t,
                real* d_tmp_dW0, real* d_tmp_dW1, int normalization)
{
    int M = y_n_rows;
    int N = y_n_cols;
    real alpha = 1.0;
    real beta = 0.0;
    GPUtranspose(d_X, d_XT, X_n_rows, X_n_cols);
    GPUtranspose(d_W1, d_W1_t, nn.W[1].n_rows, nn.W[1].n_cols);

    cudaMemcpy(d_tmp_dW1, d_W1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tmp_dW0, d_W0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToDevice);

    GPUaddition(d_yc, d_y, d_diff, 1.0/normalization, -1.0/normalization, M, N);

    GPUtranspose(d_a1, d_a1_t, nn.b[0].n_rows, X_n_cols);
    
    myGEMM(d_diff, d_a1_t, d_tmp_dW1, &alpha, &reg, nn.W[1].n_rows, nn.W[1].n_cols, N);
    GPUrowSum(d_diff, d_tmp_db1, M, N);

    myGEMM(d_W1_t, d_diff, d_da1, &alpha, &beta, nn.W[1].n_cols, N, nn.W[1].n_rows);
    GPUmult(d_da1, d_a1, d_tmp1, nn.W[1].n_cols, N);
    GPUmult(d_tmp1, d_a1, d_tmp2, nn.W[1].n_cols, N);
    GPUaddition(d_tmp1, d_tmp2, d_dz1, 1.0, -1.0, nn.W[1].n_cols, N);

    myGEMM(d_dz1, d_XT, d_tmp_dW0, &alpha, &reg, nn.W[0].n_rows, nn.W[0].n_cols, N);
    GPUrowSum(d_dz1, d_tmp_db0, nn.W[1].n_cols, N);
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<real> &X, const arma::Mat<real> &y,
                    real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug)
{

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0) ? X.n_cols : 0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    int iter = 0;
    int X_n_rows;
    int X_n_cols;
    int y_n_rows;
    X_n_rows = X.n_rows; // 784
    X_n_cols = X.n_cols; // 54,000
    y_n_rows = y.n_rows; // 10
    
    std::cout << "** Dimensions: " << X_n_rows << " " << X_n_cols << " " << y_n_rows << " " << y.n_cols << "\n";
    MPI_SAFE_CALL(MPI_Bcast(&X_n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&X_n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&y_n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));

    int num_batches = (N + batch_size - 1) / batch_size; 
    int minibatch_size = batch_size / num_procs;
    minibatch_size = rank < batch_size % num_procs ? minibatch_size + 1 : minibatch_size;
    std::cout << "** rank:" << rank << ", minibatch_size: " << minibatch_size << "\n";

    std::vector<arma::Mat<real>> X_batches;
    std::vector<arma::Mat<real>> y_batches;

    int *displs_x = new int[num_procs];
    int *displs_y = new int[num_procs];
    int *counts_x = new int[num_procs];
    int *counts_y = new int[num_procs];

    arma::Mat<real> my_X(X_n_rows, minibatch_size);
    arma::Mat<real> my_y(y_n_rows, minibatch_size);
    for (int batch = 0; batch < num_batches-1; ++batch) {
        arma::Mat<real> X_batch(X_n_rows, batch_size);
        arma::Mat<real> y_batch(y_n_rows, batch_size);
        if (rank==0) {
            int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
            X_batch = X.cols(batch * batch_size, last_col);
            y_batch = y.cols(batch * batch_size, last_col);
        }
        MPI_SAFE_CALL(MPI_Scatter(
          X_batch.memptr(), X_n_rows * minibatch_size, MPI_FP, my_X.memptr(), X_n_rows * minibatch_size, MPI_FP, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Scatter(
          y_batch.memptr(), y_n_rows * minibatch_size, MPI_FP, my_y.memptr(), y_n_rows * minibatch_size, MPI_FP, 0, MPI_COMM_WORLD));
        X_batches.push_back(my_X);
        y_batches.push_back(my_y);
    }
    
    for (int i = 0; i < num_procs; i++) {
        int last_batch_size = X_n_cols - batch_size * (num_batches - 1);
        int last_minibatch_size = (last_batch_size + num_procs - 1) / num_procs;
        displs_x[i] = X_n_rows * last_minibatch_size * i + X_n_rows * (num_batches - 1) * batch_size;
        displs_y[i] = y_n_rows * last_minibatch_size * i + y_n_rows * (num_batches - 1) * batch_size;
        counts_x[i] = X_n_rows * last_minibatch_size;
        counts_y[i] = y_n_rows * last_minibatch_size;
    }

    arma::Mat<real> last_X(X_n_rows, counts_x[rank] / X_n_rows);
    arma::Mat<real> last_y(y_n_rows, counts_y[rank] / y_n_rows);
    MPI_Scatterv(X.memptr(), counts_x, displs_x, MPI_FP, last_X.memptr(), counts_x[rank], MPI_FP, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y.memptr(), counts_y, displs_y, MPI_FP, last_y.memptr(), counts_y[rank], MPI_FP, 0, MPI_COMM_WORLD);
    X_batches.push_back(last_X);
    y_batches.push_back(last_y);

    real *dW0 = new real[nn.W[0].n_rows * nn.W[0].n_cols];
    real *dW1 = new real[nn.W[1].n_rows * nn.W[1].n_cols];
    real *db0 = new real[nn.b[0].n_rows * nn.b[0].n_cols];
    real *db1 = new real[nn.b[1].n_rows * nn.b[1].n_cols];
    real *host_dW1 = new real[nn.W[1].n_rows * nn.W[1].n_cols];
    real *host_dW0 = new real[nn.W[0].n_rows * nn.W[0].n_cols];
    real *host_db1 = new real[nn.b[1].n_rows * nn.b[1].n_cols];
    real *host_db0 = new real[nn.b[0].n_rows * nn.b[0].n_cols];
    real *d_z1;
    real *d_z2;
    real *d_yc;
    real *d_a1;
    real *d_X;
    real *d_XT;
    real *d_W0;
    real *d_W1;
    real *d_dW0;
    real *d_dW1;
    real *d_W1_t;
    real *d_b0;
    real *d_b1;
    real *d_db0;
    real *d_db1;
    real *d_y;
    real *d_diff;
    real *d_da1;
    real *d_tmp1;
    real *d_tmp2;
    real *d_dz1;
    real *d_a1_t;
    real *d_tmp_dW1;
    real *d_tmp_dW0;
    real *d_tmp_db1;
    real *d_tmp_db0;
    real* d_exp;
    cudaMalloc((void**)&d_exp, sizeof(real) * minibatch_size);
    cudaMalloc((void **)&d_tmp_dW1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_tmp_dW0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_tmp_db1, sizeof(real) * y_n_rows * 1);
    cudaMalloc((void **)&d_tmp_db0, sizeof(real) * nn.W[1].n_cols * 1);
    cudaMalloc((void **)&d_da1, sizeof(real) * nn.W[1].n_cols * minibatch_size);
    cudaMalloc((void **)&d_tmp1, sizeof(real) * nn.W[1].n_cols * minibatch_size);
    cudaMalloc((void **)&d_tmp2, sizeof(real) * nn.W[1].n_cols * minibatch_size);
    cudaMalloc((void **)&d_dz1, sizeof(real) * nn.W[1].n_cols * minibatch_size);
    cudaMalloc((void **)&d_a1_t, sizeof(real) * X_n_cols * nn.b[0].n_rows);
    cudaMalloc((void **)&d_z1, sizeof(real) * nn.b[0].n_rows * N);
    cudaMalloc((void **)&d_z2, sizeof(real) * nn.b[1].n_rows * N);
    cudaMalloc((void **)&d_yc, sizeof(real) * nn.b[1].n_rows * N);
    cudaMalloc((void **)&d_a1, sizeof(real) * nn.b[0].n_rows * minibatch_size);
    cudaMalloc((void **)&d_X, sizeof(real) * X_n_rows * minibatch_size);
    cudaMalloc((void **)&d_XT, sizeof(real) * minibatch_size * X_n_rows);
    cudaMalloc((void **)&d_W0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_W1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_dW0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols);
    cudaMalloc((void **)&d_dW1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols);
    cudaMalloc((void **)&d_W1_t, sizeof(real) * nn.W[1].n_cols * nn.W[1].n_rows);
    cudaMalloc((void **)&d_b0, sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols);
    cudaMalloc((void **)&d_b1, sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols);
    cudaMalloc((void **)&d_db0, sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols);
    cudaMalloc((void **)&d_db1, sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols);
    cudaMalloc((void **)&d_y, sizeof(real) * y_n_rows * minibatch_size);
    cudaMalloc((void **)&d_diff, sizeof(real) * y_n_rows * minibatch_size);
    cudaMemcpy(d_W0, nn.W[0].memptr(), sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, nn.b[0].memptr(), sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, nn.b[1].memptr(), sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyHostToDevice);


    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (int batch = 0; batch < num_batches; ++batch)
        {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */

            real *X_batch_memptr = X_batches[batch].memptr();
            real *y_batch_memptr = y_batches[batch].memptr();
            int X_batch_n_rows = X_batches[batch].n_rows;
            int X_batch_n_cols = X_batches[batch].n_cols;
            int y_batch_n_rows = y_batches[batch].n_rows;
            int y_batch_n_cols = y_batches[batch].n_cols;
            int normalization;
            if (batch == num_batches-1) {
                normalization = X_n_cols - (num_batches - 1) * batch_size;
            } else {
                normalization = batch_size;
            }
                
            real reg_adj = reg / num_procs;
          
            cudaMemcpy(d_X, X_batch_memptr, sizeof(real) * X_batch_n_rows * X_batch_n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y_batch_memptr, sizeof(real) * y_n_rows * minibatch_size, cudaMemcpyHostToDevice);
           
            GPUfeedforward(nn, d_X, X_batch_n_rows, X_batch_n_cols, d_W0, d_W1, d_b0, d_b1, d_a1, d_yc, d_z1, d_z2, d_exp);
            GPUbackprop(nn, d_y, d_diff, d_yc, y_batch_n_rows, y_batch_n_cols, reg_adj, d_X, d_XT, X_batch_n_rows, X_batch_n_cols, d_a1,
                d_W0, d_W1, d_W1_t, d_tmp_db1, d_da1, d_tmp1, d_tmp2, d_dz1, d_tmp_db0, d_a1_t, d_tmp_dW0, d_tmp_dW1, 
                normalization);
            
            cudaMemcpy(host_dW0, d_tmp_dW0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToHost);
            MPI_SAFE_CALL(MPI_Allreduce(host_dW0, dW0, nn.W[0].n_rows * nn.W[0].n_cols, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

            cudaMemcpy(host_dW1, d_tmp_dW1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyDeviceToHost);
            MPI_SAFE_CALL(MPI_Allreduce(host_dW1, dW1, nn.W[1].n_rows * nn.W[1].n_cols, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

            cudaMemcpy(host_db0, d_tmp_db0, sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyDeviceToHost);
            MPI_SAFE_CALL(MPI_Allreduce(host_db0, db0, nn.b[0].n_rows * nn.b[0].n_cols, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

            cudaMemcpy(host_db1, d_tmp_db1, sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyDeviceToHost);
            MPI_SAFE_CALL(MPI_Allreduce(host_db1, db1, nn.b[1].n_rows * nn.b[1].n_cols, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

            cudaMemcpy(d_dW0, dW0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dW1, dW1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_db0, db0, sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_db1, db1, sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyHostToDevice);

            bool copy_flag = (epoch == epochs -1 && batch == num_batches - 1);
       
            GPUaddition(d_W0, d_dW0, d_W0, 1.0, -learning_rate, nn.W[0].n_rows, nn.W[0].n_cols);
            if (copy_flag) cudaMemcpy(nn.W[0].memptr(), d_W0, sizeof(real) * nn.W[0].n_rows * nn.W[0].n_cols, cudaMemcpyDeviceToHost);
            GPUaddition(d_W1, d_dW1, d_W1, 1.0, -learning_rate, nn.W[1].n_rows, nn.W[1].n_cols);
            if (copy_flag) cudaMemcpy(nn.W[1].memptr(), d_W1, sizeof(real) * nn.W[1].n_rows * nn.W[1].n_cols, cudaMemcpyDeviceToHost);
            GPUaddition(d_b0, d_db0, d_b0, 1.0, -learning_rate, nn.b[0].n_rows, nn.b[0].n_cols);
            if (copy_flag) cudaMemcpy(nn.b[0].memptr(), d_b0, sizeof(real) * nn.b[0].n_rows * nn.b[0].n_cols, cudaMemcpyDeviceToHost);
            GPUaddition(d_b1, d_db1, d_b1, 1.0, -learning_rate, nn.b[1].n_rows, nn.b[1].n_cols);
            if (copy_flag) cudaMemcpy(nn.b[1].memptr(), d_b1, sizeof(real) * nn.b[1].n_rows * nn.b[1].n_cols, cudaMemcpyDeviceToHost);

            if (print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if (debug && rank == 0 && print_flag)
            {
                write_diff_gpu_cpu(nn, iter, error_file);
            }
            iter++;
        }
    }

    error_file.close();

    // free memory
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_W1_t);
    cudaFree(d_dW0);
    cudaFree(d_dW1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    cudaFree(d_db0);
    cudaFree(d_db1);
    cudaFree(d_a1);
    cudaFree(d_X);
    cudaFree(d_XT);
    cudaFree(d_y);
    cudaFree(d_diff);
    cudaFree(d_tmp_db1);
    cudaFree(d_da1);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    cudaFree(d_dz1);
    cudaFree(d_tmp_db0);
    cudaFree(d_a1_t);
    cudaFree(d_tmp_dW1);
    cudaFree(d_tmp_dW0);
    cudaFree(d_exp);

    delete[] host_dW0;
    delete[] host_dW1;
    delete[] host_db0;
    delete[] host_db1;
    delete[] displs_x;
    delete[] displs_y;
    delete[] counts_x;
    delete[] counts_y;
    delete[] dW0;
    delete[] dW1;
    delete[] db0;
    delete[] db1;
}
