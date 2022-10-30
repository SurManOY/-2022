#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#define TARGET M_PI/24.0
#define V 4.0
int main(int argc, char *argv[]) 
{
    int i, rank, size, iter = 1, stop = 1;
    double xy, x, y, z, process_sum = 0.0, total_sum = 0.0, result = 0.0,eps = std::atof(argv[1]); 
    int seed = std::atoi(argv[2]);
    double result_time, time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int batch = 80.0/(size*eps);
    batch *= 16;
    drand48_data rand_array[16];
    int block_r = size*batch/16;
    for(i = 0; i < 16; ++i)
        srand48_r(seed+i, &rand_array[i]);
    
    double start_time = MPI_Wtime();
    while(stop){
      process_sum = 0.0;
      for(i = 0; i < batch; ++i)
      {
        drand48_r(&rand_array[(i/block_r)*size + rank], &x);
        drand48_r(&rand_array[(i/block_r)*size + rank], &y);
        drand48_r(&rand_array[(i/block_r)*size + rank], &z);
        xy = x*x + y*y;
        if (xy <= z*z)
          process_sum += sqrt(xy);
      }
      MPI_Allreduce(&process_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      result += total_sum/(size*batch); 
      if (std::fabs(TARGET - result/iter) < eps){
        stop = 0;
      }
      ++iter;
    }
    double end_time = MPI_Wtime();
    
    time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "Integral: " << V*result/(iter-1) << std::endl;
      std::cout << "Eps: " << std::abs(TARGET - result/(iter-1)) << std::endl;
      std::cout << "N points: " << size * batch * (iter-1) << std::endl;
      std::cout << "Time: " << result_time << std::endl;
      std::cout << "Iters; " << iter << std::endl;
      std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}
