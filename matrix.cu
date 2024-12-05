#include "matrix.h"

//
// Constructors of Matrix
//

// Default constructor
template <typename T>
Matrix<T>::Matrix()
{
    data = nullptr;
    row = 0;
    col = 0;
}

// Initialize a all-zero matrix
template <typename T>
Matrix<T>::Matrix(ll r, ll c)
{
    row = r;
    col = c;
    data = new T *[row];
    for (ll i = 0; i < row; i++)
    {
        data[i] = new T[col];
        for (ll j = 0; j < col; j++)
        {
            data[i][j] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

// Initialize a matrix with a given 2D array
template <typename T>
Matrix<T>::Matrix(ll r, ll c, T **temp)
{
    row = r;
    col = c;
    data = new T *[row];
    for (ll i = 0; i < row; i++)
    {
        data[i] = new T[col];
        for (ll j = 0; j < col; j++)
        {
            data[i][j] = *((T *)temp + i * col + j);
        }
    }
}

// Copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T> &matrx)
{
    row = matrx.row;
    col = matrx.col;
    data = new T *[row];
    for (ll i = 0; i < row; i++)
    {
        data[i] = new T[col];
        memcpy(data[i], matrx.data[i], col * sizeof(T));
    }
}

// Move constructor
template <typename T>
Matrix<T>::Matrix(Matrix<T> &&matrx)
{
    row = matrx.row;
    col = matrx.col;
    data = matrx.data;
    matrx.row = 0;
    matrx.col = 0;
    matrx.data = nullptr;
}

// 为对象在设备上分配内存的成员函数
template <typename T>
cudaError_t Matrix<T>::allocateDeviceMemory(Matrix<DTYPE> *&deviceMatrix, const Matrix<DTYPE> &hostMatrix)
{

    // 1. 分配 deviceMatrix 对象内存
    HANDLE_CUDA_ERROR(cudaMalloc(&deviceMatrix, sizeof(Matrix<T>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(deviceMatrix, &hostMatrix, sizeof(Matrix<T>), cudaMemcpyHostToDevice));
    // 2. 分配 data 指针数组内存
    T **deviceData;
    HANDLE_CUDA_ERROR(cudaMalloc(&deviceData, hostMatrix.row * sizeof(T *)));

    // 3. 分配每一行数据内存并将指针复制到 deviceData
    for (ll i = 0; i < hostMatrix.row; ++i)
    {
        T *rowPtr;
        HANDLE_CUDA_ERROR(cudaMalloc(&rowPtr, hostMatrix.col * sizeof(T)));
        HANDLE_CUDA_ERROR(cudaMemcpy(deviceData + i, &rowPtr, sizeof(T *), cudaMemcpyHostToDevice));
    }

    // 4. 将 deviceData 指针复制到 deviceMatrix
    HANDLE_CUDA_ERROR(cudaMemcpy(&(deviceMatrix->data), &deviceData, sizeof(T **), cudaMemcpyHostToDevice));

    // 5. 复制行列信息
    HANDLE_CUDA_ERROR(cudaMemcpy(&(deviceMatrix->row), &(hostMatrix.row), sizeof(ll), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(&(deviceMatrix->col), &(hostMatrix.col), sizeof(ll), cudaMemcpyHostToDevice));

    // 6. 复制数据
    for (ll i = 0; i < hostMatrix.row; ++i)
    {
        T *rowPtr;
        HANDLE_CUDA_ERROR(cudaMemcpy(&rowPtr, deviceData + i, sizeof(T *), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaMemcpy(rowPtr, hostMatrix.data[i], hostMatrix.col * sizeof(T), cudaMemcpyHostToDevice));
    }
    return cudaSuccess; // 返回成功状态
}

template <typename T>
cudaError_t Matrix<T>::copyDeviceToHost(Matrix<DTYPE> *deviceMatrix, Matrix<DTYPE> &hostMatrix)
{
    T **deviceData;
    HANDLE_CUDA_ERROR(cudaMemcpy(&deviceData, &(deviceMatrix->data), sizeof(T **), cudaMemcpyDeviceToHost));

    // 复制每一行的数据
    for (ll i = 0; i < hostMatrix.row; ++i)
    {
        T *rowPtr;
        HANDLE_CUDA_ERROR(cudaMemcpy(&rowPtr, deviceData + i, sizeof(T *), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaMemcpy(hostMatrix.data[i], rowPtr, hostMatrix.col * sizeof(T), cudaMemcpyDeviceToHost));
    }

    return cudaSuccess; // 返回成功状态
}

template <typename T>
cudaError_t Matrix<T>::copyHostToDevice(Matrix<DTYPE> &hostMatrix, Matrix<DTYPE> *deviceMatrix)
{

    // 1. 更新设备上的行列信息
    HANDLE_CUDA_ERROR(cudaMemcpy(&(deviceMatrix->row), &(hostMatrix.row), sizeof(ll), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(&(deviceMatrix->col), &(hostMatrix.col), sizeof(ll), cudaMemcpyHostToDevice));

    // 2. 获取设备上的data指针
    T **deviceData;
    HANDLE_CUDA_ERROR(cudaMemcpy(&deviceData, &(deviceMatrix->data), sizeof(T **), cudaMemcpyDeviceToHost));

    // 3. 复制实际数据
    for (ll i = 0; i < hostMatrix.row; ++i)
    {
        T *rowPtr;
        HANDLE_CUDA_ERROR(cudaMemcpy(&rowPtr, deviceData + i, sizeof(T *), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaMemcpy(rowPtr, hostMatrix.data[i], hostMatrix.col * sizeof(T), cudaMemcpyHostToDevice));
    }
    return cudaSuccess;
}
// 释放设备内存
template <typename T>
cudaError_t Matrix<T>::freeDeviceMemory(Matrix<DTYPE> *deviceMatrix)
{
    // 1. 检查 deviceMatrix 是否为空
    if (deviceMatrix == nullptr)
    {
        return cudaSuccess; // 如果为空，则无需释放，直接返回成功
    }

    // 2. 获取 deviceData 指针
    T **deviceData;
    HANDLE_CUDA_ERROR(cudaMemcpy(&deviceData, &(deviceMatrix->data), sizeof(T **), cudaMemcpyDeviceToHost));

    ll hostRow;
    HANDLE_CUDA_ERROR(cudaMemcpy(&hostRow, &(deviceMatrix->row), sizeof(ll), cudaMemcpyDeviceToHost));

    // 3. 释放每一行的数据内存
    for (ll i = 0; i < hostRow; ++i)
    {
        T *rowPtr;
        HANDLE_CUDA_ERROR(cudaMemcpy(&rowPtr, deviceData + i, sizeof(T *), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(rowPtr));
    }

    // 4. 释放 deviceData 指针数组内存
    HANDLE_CUDA_ERROR(cudaFree(deviceData));

    // 5. 释放 deviceMatrix 对象内存
    HANDLE_CUDA_ERROR(cudaFree(deviceMatrix));

    return cudaSuccess; // 返回成功状态
}
//
// Operations
//

template <typename T>
void Matrix<T>::clear()
{
    if (data != nullptr)
    {
        for (ll i = 0; i < row; i++)
        {
            delete[] data[i];
        }
        delete[] data;
        data = nullptr;
    }
    row = 0;
    col = 0;
}

// Copy assignment
template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &matrx)
{
    if (this != &matrx)
    {
        clear();
        row = matrx.row;
        col = matrx.col;
        data = new T *[row];
        for (ll i = 0; i < row; i++)
        {
            data[i] = new T[col];
            memcpy(data[i], matrx.data[i], col * sizeof(T));
        }
    }
    return *this;
}

// Move assignment
template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&matrx)
{
    if (this != &matrx)
    {
        clear();
        row = matrx.row;
        col = matrx.col;
        data = matrx.data;
        matrx.row = 0;
        matrx.col = 0;
        matrx.data = nullptr;
    }
    return *this;
}

// Matrix addition C = A + B
template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &matrx) const
{
    if (row != matrx.row || col != matrx.col)
    {
        printf("[ERROR] Matrix +: row(%lld) != matrx.row(%lld) || col(%lld) != matrx.col(%lld).\n", row, matrx.row, col, matrx.col);
        return *this;
    }
    Matrix<T> temp(row, col);
    for (ll i = 0; i < row; i++)
    {
        for (ll j = 0; j < col; j++)
        {
            temp.data[i][j] = cuCadd(data[i][j], matrx.data[i][j]);
        }
    }
    return temp;
}

// Matrix addition A += B
template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &matrx)
{
    if (row != matrx.row || col != matrx.col)
    {
        printf("[ERROR] Matrix +=: row(%lld) != matrx.row(%lld) || col(%lld) != matrx.col(%lld). ", row, matrx.row, col, matrx.col);
        return *this;
    }
    for (ll i = 0; i < row; i++)
    {
        for (ll j = 0; j < col; j++)
        {
            data[i][j] = cuCadd(data[i][j], matrx.data[i][j]);
        }
    }
    return *this;
}

// Matrix multiplication C = A * B
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &matrx) const
{
    if (col != matrx.row)
    {
        printf("[ERROR] Matrix *: col != matrx.row.");
        return *this;
    }
    Matrix<T> temp(row, matrx.col);
    for (ll i = 0; i < row; i++)
    {
        for (ll j = 0; j < matrx.col; j++)
        {
            for (ll k = col - 1; k >= 0; k--)
            {
                temp.data[i][j] = cuCadd(cuCmul(data[i][k], matrx.data[k][j]), temp.data[i][j]);
            }
        }
    }
    return temp;
}

// Tensor product C = A tensorProduct B
template <typename T>
Matrix<T> Matrix<T>::tensorProduct(const Matrix<T> &matrx) const
{
    Matrix<T> temp(row * matrx.row, col * matrx.col);
    for (ll ar = 0; ar < row; ar++)
    {
        for (ll ac = 0; ac < col; ac++)
        {
            // 检查复数是否为零
            if (cuCabs(data[ar][ac]) < 1e-10)
                continue;
            for (ll br = 0; br < matrx.row; br++)
            {
                for (ll bc = 0; bc < matrx.col; bc++)
                {
                    // 使用 cuComplex 乘法
                    temp.data[ar * matrx.row + br][ac * matrx.col + bc] = cuCmul(data[ar][ac], matrx.data[br][bc]);
                }
            }
        }
    }
    return temp;
}

// Rotation X
template <typename T>
void Matrix<T>::rotationX(double theta)
{
    T rx[2][2];

    // 计算 sin 和 cos 值
    double cosTheta = cos(theta / 2);
    double sinTheta = sin(theta / 2);

    // 使用 make_cuDoubleComplex 函数创建复数
    rx[0][0] = make_cuDoubleComplex(cosTheta, 0.0);
    rx[0][1] = make_cuDoubleComplex(0.0, -sinTheta);
    rx[1][0] = make_cuDoubleComplex(0.0, -sinTheta);
    rx[1][1] = make_cuDoubleComplex(cosTheta, 0.0);
    clear();
    row = 2;
    col = 2;
    data = new T *[2];
    for (ll i = 0; i < 2; i++)
    {
        data[i] = new T[2];
        for (ll j = 0; j < 2; j++)
        {
            data[i][j] = rx[i][j];
        }
    }
}

// Rotation Y
template <typename T>
void Matrix<T>::rotationY(double theta)
{
    T ry[2][2];

    // 计算 sin 和 cos 值
    double cosTheta = cos(theta / 2);
    double sinTheta = sin(theta / 2);

    // 使用 make_cuDoubleComplex 函数创建复数
    ry[0][0] = make_cuDoubleComplex(cosTheta, 0.0);
    ry[0][1] = make_cuDoubleComplex(-sinTheta, 0.0);
    ry[1][0] = make_cuDoubleComplex(sinTheta, 0.0);
    ry[1][1] = make_cuDoubleComplex(cosTheta, 0.0);

    clear();
    row = 2;
    col = 2;
    data = new T *[2];
    for (ll i = 0; i < 2; i++)
    {
        data[i] = new T[2];
        for (ll j = 0; j < 2; j++)
        {
            data[i][j] = ry[i][j];
        }
    }
}

// Rotation Z
template <typename T>
void Matrix<T>::rotationZ(double theta)
{
    T rz[2][2];

    // 计算 exp(±iθ/2)
    double cosHalfTheta = cos(theta / 2);
    double sinHalfTheta = sin(theta / 2);

    // 使用 make_cuDoubleComplex 函数创建复数
    rz[0][0] = make_cuDoubleComplex(cosHalfTheta, -sinHalfTheta); // exp(-iθ/2)
    rz[0][1] = make_cuDoubleComplex(0.0, 0.0);
    rz[1][0] = make_cuDoubleComplex(0.0, 0.0);
    rz[1][1] = make_cuDoubleComplex(cosHalfTheta, sinHalfTheta); // exp(iθ/2)

    clear();
    row = 2;
    col = 2;
    data = new T *[2];
    for (ll i = 0; i < 2; i++)
    {
        data[i] = new T[2];
        for (ll j = 0; j < 2; j++)
        {
            data[i][j] = rz[i][j];
        }
    }
}

// Set the matrix to be an identity matrix
template <typename T>
void Matrix<T>::identity(ll r)
{
    clear();
    row = r;
    col = r;
    data = new T *[row];
    for (ll i = 0; i < row; i++)
    {
        data[i] = new T[col];
        for (ll j = 0; j < col; j++)
        {
            data[i][j] = (i == j) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

// Set the matrix to be a zero matrix
template <typename T>
void Matrix<T>::zero(ll r, ll c)
{
    clear();
    row = r;
    col = c;
    data = new T *[row];
    for (ll i = 0; i < row; i++)
    {
        data[i] = new T[col];
        for (ll j = 0; j < col; j++)
        {
            data[i][j] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

// Check if the matrix is a zero matrix
template <typename T>
bool Matrix<T>::isZero() const
{
    for (ll i = 0; i < row; ++i)
    {
        for (ll j = 0; j < col; ++j)
        {
            if (cuCreal(data[i][j]) != 0.0 || cuCimag(data[i][j]) != 0.0)
            {
                return false;
            }
        }
    }
    return true;
}

//
// Utility functions
//

// Print the matrix
template <typename T>
void Matrix<T>::print() const
{
    cout << "----- Matrix: [" << row << "] * [" << col << "] -----" << endl;
    cout.setf(std::ios::left);
    if (data == nullptr)
    {
        cout << "Matrix is Empty!" << endl;
    }
    else
    {
        for (ll i = 0; i < row; i++)
        {
            for (ll j = 0; j < col; j++)
            {
                cout.width(13);
                cout << fixed << setprecision(4) << '(' << data[i][j].x << ',' << data[i][j].y << ')';
            }
            cout << endl;
        }
    }
}

// Print the matrix dictionary
template <typename T>
void Matrix<T>::printMatrixDict()
{
    for (auto it = MatrixDict.begin(); it != MatrixDict.end(); ++it)
    {
        cout << it->first << ": " << endl;
        it->second->print();
    }
}

// 将矩阵数据写入文本文件
template <typename T>
void Matrix<T>::writeToTextFile(string filename)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        throw runtime_error("Error opening file: " + filename);
    }

    // 写入矩阵维度
    // file << row << " " << col << endl;
    // 设置输出格式为保留小数点后4位
    file << fixed << setprecision(8);
    // 写入矩阵数据
    for (ll i = 0; i < row; ++i)
    {
        for (ll j = 0; j < col; ++j)
        {
            file << data[i][j].x << ' ' << data[i][j].y << (j == col - 1 ? "" : " "); // 最后一个元素后不加空格
        }
        file << endl;
    }

    if (file.fail())
    {
        throw runtime_error("Error writing to file: " + filename);
    }

    file.close();
}

//
// Destructor
//
template <typename T>
Matrix<T>::~Matrix()
{
    clear();
    // cout << "~Matrix Class Destruct!" << endl;
}

//
// Global notations
//

template <typename T>
map<string, shared_ptr<Matrix<T>>> Matrix<T>::MatrixDict; // A global matrix dictionary

template <typename T>
void Matrix<T>::initMatrixDict()
{
    // T mark[1][1] = {{1}}; // placeholder
    // MatrixDict["MARK"] = make_shared<Matrix<T>>(1, 1, (T**)mark);

    // T zeros[2][1] = {{1}, {0}};
    // MatrixDict["ZEROS"] = make_shared<Matrix<T>>(2, 1, (T**)zeros);

    // T ones[2][1] = {{0}, {1}};
    // MatrixDict["ONES"] = make_shared<Matrix<T>>(2, 1, (T**)ones);

    // T plus[2][1] = {{1.0 / sqrt(2)}, {1.0 / sqrt(2)}};
    // MatrixDict["PLUS"] = make_shared<Matrix<T>>(2, 1, (T**)plus);

    // T minus[2][1] = {{1.0 / sqrt(2)}, {-1.0 / sqrt(2)}};
    // MatrixDict["MINUS"] = make_shared<Matrix<T>>(2, 1, (T**)minus);

    // -------------- Gates -----------------

    T ide[2][2] = {
        {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)}};
    MatrixDict["IDE"] = make_shared<Matrix<cuDoubleComplex>>(2, 2, (cuDoubleComplex **)ide);
    MatrixDict["MARK"] = MatrixDict["IDE"];

    // H 矩阵
    double invSqrt2 = 1.0 / std::sqrt(2.0);
    T h[2][2] = {
        {make_cuDoubleComplex(invSqrt2, 0.0), make_cuDoubleComplex(invSqrt2, 0.0)},
        {make_cuDoubleComplex(invSqrt2, 0.0), make_cuDoubleComplex(-invSqrt2, 0.0)}};
    MatrixDict["H"] = make_shared<Matrix<cuDoubleComplex>>(2, 2, (cuDoubleComplex **)h);

    // X 矩阵
    T x[2][2] = {
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)},
        {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)}};
    MatrixDict["X"] = make_shared<Matrix<cuDoubleComplex>>(2, 2, (cuDoubleComplex **)x);
    MatrixDict["CX"] = MatrixDict["X"];

    // Y 矩阵
    T y[2][2] = {
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, -1.0)},
        {make_cuDoubleComplex(0.0, 1.0), make_cuDoubleComplex(0.0, 0.0)}};
    MatrixDict["Y"] = make_shared<Matrix<cuDoubleComplex>>(2, 2, (cuDoubleComplex **)y);
    MatrixDict["CY"] = MatrixDict["Y"];

    // Z 矩阵
    T z[2][2] = {
        {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(-1.0, 0.0)}};
    MatrixDict["Z"] = make_shared<Matrix<cuDoubleComplex>>(2, 2, (cuDoubleComplex **)z);
    MatrixDict["CZ"] = MatrixDict["Z"];

    // SWAP 矩阵
    T swap[4][4] = {
        {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
        {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)}};
    MatrixDict["SWAP"] = make_shared<Matrix<T>>(4, 4, (T **)swap);
    MatrixDict["CSWAP"] = MatrixDict["SWAP"];
}

template class Matrix<DTYPE>;
