#pragma once

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include "helper.hpp"
using namespace std;

#define ll long long int
#define DTYPE cuDoubleComplex

template <typename T>
class Matrix
{
private:
    void clear(); // Clear the matrix
public:
    ll row, col;
    T **data;
    // static map<string, shared_ptr<Matrix>> MatrixDict; // A global matrix dictionary

    //
    // Constructors
    //
    Matrix();           // Default constructor
    Matrix(ll r, ll c); // Initialize a all-zero matrix
    Matrix(ll r, ll c, T **temp);
    Matrix(const Matrix &matrx); // Copy constructor
    Matrix(Matrix &&matrx);      // Move constructor

    // static cudaError_t allocateDeviceMemory(Matrix<DTYPE> *&deviceMatrix, const Matrix<DTYPE> &hostMatrix);
    // static cudaError_t copyDeviceToHost(Matrix<DTYPE> *deviceMatrix, Matrix<DTYPE> &hostMatrix);
    // static cudaError_t freeDeviceMemory(Matrix<DTYPE> *deviceMatrix);
    // static cudaError_t copyHostToDevice(Matrix<DTYPE> &hostMatrix, Matrix<DTYPE> *deviceMatrix);
    
    //
    // Operations
    //
    Matrix &operator=(const Matrix &matrx); // Copy assignment
    Matrix &operator=(Matrix &&matrx);      // Move assignment

    Matrix operator+(const Matrix &matrx) const;     // Matrix addition
    Matrix &operator+=(const Matrix &matrx);         // Matrix addition
    Matrix operator*(const Matrix &matrx) const;     // Matrix multiplication
    Matrix tensorProduct(const Matrix &matrx) const; // Tensor product

    void rotationX(double theta); // Rotation X gate matrix
    void rotationY(double theta); // Rotation Y gate matrix
    void rotationZ(double theta); // Rotation Z gate matrix

    void identity(ll r);   // Set the matrix to be an identity matrix
    void zero(ll r, ll c); // Set the matrix to be a zero matrix

    bool isZero() const; // Check if the matrix is a zero matrix

    //
    // Utility functions
    //
    void print() const;     // Print a Matrix
    void printMatrixDict(); // Print the matrix dictionary
    void writeToTextFile(string filename);
    static void initMatrixDict();

    //
    // D
    //
    ~Matrix();
};