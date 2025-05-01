// g++ -Wall -o produce_HPL.exe produce_HPL.cpp -std=c++17 -stdlib=libc++ `adani-config --cppflags --ldflags --cxxflags`
#include "adani/adani.h"

#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <array>

using namespace std;

uint HPL_map(int id0 = -1, int id1 = -1, int id2 = -1, int id3 = -1, int id4 = -1) {
    return (id0 + 1) + (id1 + 1) * 3 + (id2 + 1) * 3 * 3 + (id3 + 1) * 3 * 3 * 3 + (id4 + 1) * 3 * 3 * 3 * 3;
}

int main() {

    ifstream inputx;
    inputx.open("HPL/HPL_x.txt");

    std::vector<double> x_;

    double xtmp;

    while (inputx >> xtmp) {
        x_.push_back(xtmp);
    }

    const int N = 39;
    std::array<std::vector<double>, N> HPL_matrix;

    for (double x : x_) {

        double wx = x;
        int nw = 5;
        int n1 = -1;
        int n2 = 1;
        int sz = n2 - n1 + 1;
        double *Hr1 = new double[sz];
        double *Hr2 = new double[sz * sz];
        double *Hr3 = new double[sz * sz * sz];
        double *Hr4 = new double[sz * sz * sz * sz];
        double *Hr5 = new double[sz * sz * sz * sz * sz];
    
        // Call polylogs
        apf_hplog_(&wx, &nw, Hr1, Hr2, Hr3, Hr4, Hr5, &n1, &n2);

        const double H0011 = Hr4[HPL_map(0, 0, 1, 1)];
        const double H0m1m1m1 = Hr4[HPL_map(0, -1, -1, -1)];
        const double H00m1m1 = Hr4[HPL_map(0, 0, -1, -1)];
        const double H01m1m1 = Hr4[HPL_map(0, 1, -1, -1)];
        const double H0m11m1 = Hr4[HPL_map(0, -1, 1, -1)];
        const double H001m1 = Hr4[HPL_map(0, 0, 1, -1)];
        const double H011m1 = Hr4[HPL_map(0, 1, 1, -1)];
        const double H0m1m11 = Hr4[HPL_map(0, -1, -1, 1)];
        const double H00m11 = Hr4[HPL_map(0, 0, -1, 1)];
        const double H01m11 = Hr4[HPL_map(0, 1, -1, 1)];
        const double H0m101 = Hr4[HPL_map(0, -1, 0, 1)];
        const double H0m111 = Hr4[HPL_map(0, -1, 1, 1)];

        const double H00m1m1m1 = Hr5[HPL_map(0, 0, -1, -1, -1)];
        const double H0m10m1m1 = Hr5[HPL_map(0, -1, 0, -1, -1)];
        const double H000m1m1 = Hr5[HPL_map(0, 0, 0, -1, -1)];
        const double H00m10m1 = Hr5[HPL_map(0, 0, -1, 0, -1)];
        const double H0010m1 = Hr5[HPL_map(0, 0, 1, 0, -1)];
        const double H0m101m1 = Hr5[HPL_map(0, -1, 0, 1, -1)];
        const double H0001m1 = Hr5[HPL_map(0, 0, 0, 1, -1)];
        const double H0m10m11 = Hr5[HPL_map(0, -1, 0, -1, 1)];
        const double H000m11 = Hr5[HPL_map(0, 0, 0, -1, 1)];
        const double H0m1m101 = Hr5[HPL_map(0, -1, -1, 0, 1)];
        const double H00m101 = Hr5[HPL_map(0, 0, -1, 0, 1)];
        const double H00101 = Hr5[HPL_map(0, 0, 1, 0, 1)];
        const double H0m1011 = Hr5[HPL_map(0, -1, 0, 1, 1)];
        const double H00011 = Hr5[HPL_map(0, 0, 0, 1, 1)];
        const double H01011 = Hr5[HPL_map(0, 1, 0, 1, 1)];
        const double H00111 = Hr5[HPL_map(0, 0, 1, 1, 1)];
        const double H0m11m1m1 = Hr5[HPL_map(0, -1, 1, -1, -1)];
        const double H0m1m11m1 = Hr5[HPL_map(0, -1, -1, 1, -1)];
        const double H0m1m1m11 = Hr5[HPL_map(0, -1, -1, -1, 1)];
        const double H00m1m11 = Hr5[HPL_map(0, 0, -1, -1, 1)];
        const double H00m11m1 = Hr5[HPL_map(0, 0, -1, 1, -1)];
        const double H00m111 = Hr5[HPL_map(0, 0, -1, 1, 1)];
        const double H001m1m1 = Hr5[HPL_map(0, 0, 1, -1, -1)];
        const double H001m11 = Hr5[HPL_map(0, 0, 1, -1, 1)];
        const double H0011m1 = Hr5[HPL_map(0, 0, 1, 1, -1)];
        const double H0m1m1m1m1 = Hr5[HPL_map(0, -1, -1, -1, -1)];
        const double H01m1m1m1 = Hr5[HPL_map(0, 1, -1, -1, -1)];

        HPL_matrix[0].push_back(H0011);
        HPL_matrix[1].push_back(H0m1m1m1);
        HPL_matrix[2].push_back(H00m1m1);
        HPL_matrix[3].push_back(H00m11);
        HPL_matrix[4].push_back(H001m1);
        HPL_matrix[5].push_back(H0m1m11);
        HPL_matrix[6].push_back(H0m11m1);
        HPL_matrix[7].push_back(H01m1m1);
        HPL_matrix[8].push_back(H0m111);
        HPL_matrix[9].push_back(H01m11);
        HPL_matrix[10].push_back(H011m1);
        HPL_matrix[11].push_back(H0m101);

        HPL_matrix[12].push_back(H00011);
        HPL_matrix[13].push_back(H00101);
        HPL_matrix[14].push_back(H00111);
        HPL_matrix[15].push_back(H01011);
        HPL_matrix[16].push_back(H0m10m1m1);
        HPL_matrix[17].push_back(H00m1m1m1);
        HPL_matrix[18].push_back(H00m10m1);
        HPL_matrix[19].push_back(H00m101);
        HPL_matrix[20].push_back(H000m1m1);
        HPL_matrix[21].push_back(H000m11);
        HPL_matrix[22].push_back(H0001m1);
        HPL_matrix[23].push_back(H0010m1);
        HPL_matrix[24].push_back(H0m1011);
        HPL_matrix[25].push_back(H0m1m101);
        HPL_matrix[26].push_back(H0m1m11m1);
        HPL_matrix[27].push_back(H0m10m11);
        HPL_matrix[28].push_back(H0m101m1);
        HPL_matrix[29].push_back(H0m11m1m1);
        HPL_matrix[30].push_back(H00m1m11);
        HPL_matrix[31].push_back(H00m11m1);
        HPL_matrix[32].push_back(H00m111);
        HPL_matrix[33].push_back(H001m1m1);
        HPL_matrix[34].push_back(H001m11);
        HPL_matrix[35].push_back(H0011m1);
        HPL_matrix[36].push_back(H0m1m1m1m1);
        HPL_matrix[37].push_back(H0m1m1m11);
        HPL_matrix[38].push_back(H01m1m1m1);


        delete[] Hr1;
        delete[] Hr2;
        delete[] Hr3;
        delete[] Hr4;
        delete[] Hr5;

    }

    std::array<string, N> names = {
//      HPL of weight 4 
        "0011",
        "0m1m1m1",
        "00m1m1",
        "00m11",
        "001m1",
        "0m1m11",
        "0m11m1",
        "01m1m1",
        "0m111",
        "01m11",
        "011m1",
        "0m101",
//      HPL of weight 5
        "00011",
        "00101",
        "00111",
        "01011",
        "0m10m1m1",
        "00m1m1m1",
        "00m10m1",
        "00m101",
        "000m1m1",
        "000m11",
        "0001m1",
        "0010m1",
        "0m1011",
        "0m1m101",
        "0m1m11m1",
        "0m10m11",
        "0m101m1",
        "0m11m1m1",
        "00m1m11",
        "00m11m1",
        "00m111",
        "001m1m1",
        "001m11",
        "0011m1",
        "0m1m1m1m1",
        "0m1m1m11",
        "01m1m1m1"
    };

    std::array<ofstream, N> files;
    for (int i = 0; i < names.size(); i++) {
        string name = "HPL/HPL_" + names[i] + ".txt";
        files[i].open(name);
        if (!files[i].is_open()) {
            cout << "Problems in opening " << name << endl;
            exit(-1);
        }
        for (int j = 0; j < x_.size() - 1; j++) {
            files[i] << HPL_matrix[i][j] << " ";
        }
        files[i] << HPL_matrix[i][x_.size() - 1];

    }

    return 0;
}