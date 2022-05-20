#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>
//#include <omp.h>

void normalize(std::vector<double> &W, int n_tot)
{
    auto norm = 0.0;
    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        norm += W[i] * W[i];
    }

    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        // W[i] = W[i] / sqrt(norm);
        //  now our vector contains squared values
        W[i] = (W[i] * W[i]) / norm;
    }
}

template <size_t N>
auto distance(const std::array<double, N> &a, const std::array<double, N> &b)
{
    auto d = 0.0;
    for (auto i = size_t{0}; i < N; ++i)
    {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(d);
}

auto compute_integral(const long int n_sweeps, const size_t n_tot, std::vector<double> const &W, std::vector<std::array<double, 3> > const &r)
{
    int i, j;
    double coulomb_U = 0.0;
    srand(time(NULL));

    //#pragma omp parallel for default(none) firstprivate(n_tot, i, j, W, r) reduction(+ \
                                                                                 : coulomb_U)
    for (auto n = size_t{0}; n < n_sweeps; ++n)
    {
        i = rand() % n_tot;
        j = rand() % n_tot;
        if (i != j)
        {
            coulomb_U += W[i] * W[j] / distance(r[i], r[j]);
            // std::cout << distance(r[i], r[j]) << "\t" << 14.3948*(n_tot / n_sweeps)*( W[i] * W[j]) / distance(r[i], r[j]) << std::endl;
        }
    }

    return coulomb_U * (n_tot * n_tot / n_sweeps);
}

int main()
{
    const long int n_sweeps = 1E10;
    int n_tot, n_tot_new, a, b, c;
    int n_size[3];
    double vecs[3][3];
    double coulomb_U, norm;

    time_t td;
    td = time(NULL);

    std::cout << "Program Wannier_Hund.x v.2.0 starts on " << ctime(&td);
    std::cout << "=====================================================================" << std::endl;

    std::cout << "N_sweeps: " << n_sweeps << std::endl;

    std::ifstream main;
    main.open("Main.xsf");
    if (!main)
    {
        std::cout << "ERROR!Cannot open file <Main.xsf>!" << std::endl;
        return 0;
    }

    main >> n_size[0] >> n_size[1] >> n_size[2];
    std::cout << "Dimensions are: " << n_size[0] << " " << n_size[1] << " " << n_size[2] << std::endl;

    n_tot = n_size[0] * n_size[1] * n_size[2];

    std::vector<double> W(n_tot);
    std::vector<double> W_new;

    std::vector<std::array<double, 3> > r(n_tot);
    std::vector<std::array<double, 3> > r_new;
    std::array<double, 3> r_c = {5, 5, 5};

    for (auto i = size_t{0}; i < 3; ++i)
    {
        main >> vecs[i][0] >> vecs[i][1] >> vecs[i][2];
    }

    std::cout << "Span_vectors are: " << std::endl;
    for (auto i = size_t{0}; i < 3; ++i)
    {
        std::cout << vecs[i][0] << " " << vecs[i][1] << " " << vecs[i][2] << std::endl;
    }

    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        main >> W[i];
    }

    std::cout << "File <Main.xsf> was  scanned  successfully" << std::endl;

    main.close();

    normalize(W, n_tot);

    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        c = i / (n_size[0] * n_size[1]);
        a = (i - (n_size[0] * n_size[1]) * c) % (n_size[0]);
        b = (i - (n_size[0] * n_size[1]) * c) / (n_size[0]);

        for (auto j = size_t{0}; j < 3; ++j)
        {
            r[i][j] = (vecs[0][j] * a) / n_size[0] + (vecs[1][j] * b) / n_size[1] + (vecs[2][j] * c) / n_size[2];
        }

        if (distance(r_c, r[i]) < 10)
        {
            W_new.push_back(W[i]);
            r_new.push_back(r[i]);
        }
    }

    n_tot_new = W_new.size();

    norm = 0.0;
    for (auto i = size_t{0}; i < n_tot_new; ++i)
    {
        norm += W_new[i];
    }

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Size reduction leads to norm: " << norm << std::endl;
    std::cout << "At the same time it reduces the length of W from " << n_tot << " to " << n_tot_new << std::endl;
    std::cout << "i.e. gain is " << n_tot / n_tot_new << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    coulomb_U = 14.3948 * compute_integral(n_sweeps, n_tot_new, W_new, r_new); // e^2/R[A] to eV;

    std::cout << "Coulomb_U: " << coulomb_U << " eV" << std::endl;

    std::cout << std::endl
              << "=====================================================================" << std::endl;

    td = time(NULL);
    std::cout << "This run was terminated on: " << ctime(&td) << std::endl;
    std::cout << "JOB DONE" << std::endl;
    std::cout << "=====================================================================" << std::endl;
}
