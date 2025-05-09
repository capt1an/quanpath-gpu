#include "qgate.h"

QGate::QGate() {
    strncpy(gname, "NULL", sizeof(gname) - 1);
    gname[sizeof(gname) - 1] = '\0';
    numControlQubits = 0;
    numTargetQubits = 0;
    gmat = nullptr;
    theta = 0;
}
/**
 * @brief Construct a new QGate::QGate object, initialize the gate matrix with the given name
 *
 * @param gname_ the gate name
 * @param controls_ control qubits
 * @param targets_ target qubits
 */
QGate::QGate(const char* gname_, const int* controls_, int num_controls, const int* targets_, int num_targets) {
    strncpy(gname, gname_, sizeof(gname) - 1);
    gname[sizeof(gname) - 1] = '\0';
    numControlQubits = num_controls;
    theta = 0;
    for (int i = 0; i < numControlQubits; ++i) {
        controlQubits[i] = controls_[i];
    }
    numTargetQubits = num_targets;
    for (int i = 0; i < numTargetQubits; ++i) {
        targetQubits[i] = targets_[i];
    }
    if (Matrix<cuDoubleComplex>::MatrixDict.count(gname)) {
        gmat = Matrix<cuDoubleComplex>::MatrixDict[gname];
    } else {
        std::cout << "[ERROR] Gate " << gname << " not found in MatrixDict" << std::endl;
        exit(1);
    }
}

/**
 * @brief Construct a new QGate::QGate object with a parameter
 *
 * @param gname_ the gate name
 * @param controls_ control qubits
 * @param targets_ target qubits
 * @param theta a parameter
 */
QGate::QGate(const char* gname_, const int* controls_, int num_controls, const int* targets_, int num_targets, double theta) {
    strncpy(this->gname, gname_, sizeof(this->gname) - 1);
    this->gname[sizeof(this->gname) - 1] = '\0';
    numControlQubits = num_controls;
    this->theta = theta;
    for (int i = 0; i < numControlQubits; ++i) {
        controlQubits[i] = controls_[i];
    }
    numTargetQubits = num_targets;
    for (int i = 0; i < numTargetQubits; ++i) {
        targetQubits[i] = targets_[i];
    }

    std::string matkey = std::string(this->gname) + std::to_string(theta);
    if (Matrix<cuDoubleComplex>::MatrixDict.count(matkey)) {
        gmat = Matrix<cuDoubleComplex>::MatrixDict[matkey];
        return;
    }

    Matrix<cuDoubleComplex> mat(0, 0); // Initialize with invalid dimensions
    if (std::string(this->gname) == "RX") {
        // Assuming rotationX is a static method in Matrix
        mat.rotationX(theta);
    }
    else if (std::string(this->gname) == "RY") {
        mat.rotationY(theta);
    }
    else if (std::string(this->gname) == "RZ") {
        mat.rotationZ(theta);
    }
    else {
        std::cout << "[ERROR] Gate " << this->gname << " not implemented for parameterized constructor" << std::endl;
        exit(1);
    }
    Matrix<cuDoubleComplex>::MatrixDict[matkey] = std::make_shared<Matrix<cuDoubleComplex>>(std::move(mat));
    gmat = Matrix<cuDoubleComplex>::MatrixDict[matkey];
}

/**
 * @brief Copy construct a new QGate::QGate object
 *
 * @param other
 */
QGate::QGate(const QGate& other) {
    strncpy(gname, other.gname, sizeof(gname) - 1);
    gname[sizeof(gname) - 1] = '\0';
    numControlQubits = other.numControlQubits;
    this->theta = other.theta;
    for (int i = 0; i < numControlQubits; ++i) {
        controlQubits[i] = other.controlQubits[i];
    }
    numTargetQubits = other.numTargetQubits;
    for (int i = 0; i < numTargetQubits; ++i) {
        targetQubits[i] = other.targetQubits[i];
    }
    gmat = other.gmat;
}

/**
 * @brief Copy assignment
 *
 * @param other
 * @return QGate&
 */
QGate& QGate::operator=(const QGate& other) {
    if (this == &other) return *this;
    strncpy(gname, other.gname, sizeof(gname) - 1);
    gname[sizeof(gname) - 1] = '\0';
    numControlQubits = other.numControlQubits;
    this->theta = other.theta;
    for (int i = 0; i < numControlQubits; ++i) {
        controlQubits[i] = other.controlQubits[i];
    }
    numTargetQubits = other.numTargetQubits;
    for (int i = 0; i < numTargetQubits; ++i) {
        targetQubits[i] = other.targetQubits[i];
    }
    gmat = other.gmat;
    return *this;
}

// // Return the number of input/output qubits of the gate
// __host__ __device__ int QGate::numQubits() const {
//     return numControlQubits + numTargetQubits;
// }

// // Return the number of control qubits of the gate
// __host__ __device__ int QGate::numControls() const {
//     return numControlQubits;
// }

// // Return the number of target qubits of the gate
// __host__ __device__ int QGate::numTargets() const {
//     return numTargetQubits;
// }


// Print the gate information (host-only)
__host__ void QGate::print() const {
    std::cout << "===== Gate: " << gname << " =====" << std::endl;
    std::cout << "Control qubits: ";
    for (int i = 0; i < numControlQubits; ++i) {
        std::cout << controlQubits[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Target qubits: ";
    for (int i = 0; i < numTargetQubits; ++i) {
        std::cout << targetQubits[i] << " ";
    }
    std::cout << std::endl;
    if (gmat) {
        gmat->print();
    } else {
        std::cout << "Gate matrix is null." << std::endl;
    }
}

// Destructor
QGate::~QGate() {
    return;
}

// Compare two integers by their absolute values
// Control qubits can be negative to denote 0-controlled
bool compareByAbsoluteValue(int a, int b) {
    return std::abs(a) < std::abs(b);
}