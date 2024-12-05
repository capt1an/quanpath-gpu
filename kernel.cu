
#include "device_launch_parameters.h"
#include "cuquanpath.h"

int numQubits = 13;
int numThreads = 32;

int numHighQubits = log2(numThreads);		  // #high-order qubits
int numLowQubits = numQubits - numHighQubits; // #low-order qubits
ll lenSv = (1 << numQubits);				  // the length of the local state vector

// Generate a separable quantum circuit
QCircuit separableqc()
{

	QCircuit qc(numQubits, "separable");

	for (int layer = 0; layer < 1000; layer++)
	{
		if (layer % 5 == 0)
			for (int i = 0; i < numQubits; i++)
			{
				if (i >= numLowQubits) // 高阶部分
				{
					qc.rx((double)(layer + 1) / 200.0, i);
				}
				else // 低阶部分
				{
					qc.ry((double)(layer + 1) / 200.0, i);
				}
			}
		else if (layer % 5 == 1)
			for (int i = 0; i < numQubits; i++)
			{
				if (i % 3 == 0)
					qc.rx((double)i / numQubits, i);
				if (i % 3 == 1)
					qc.ry((double)i / numQubits, i);
				if (i % 3 == 2)
					qc.rz((double)i / numQubits, i);
			}
		else if (layer % 5 == 2)
		{
			for (int i = numQubits - 1; i > numLowQubits; i -= 2)
			{
				qc.cx(i, i - 1);
			}
			for (int i = 1; i < numLowQubits; i += 2)
			{
				qc.cy(i, i - 1);
			}
		}
		else if (layer % 5 == 3)
		{
			for (int i = numQubits - 1; i > numLowQubits; i -= 2)
			{
				qc.swap(i, i - 1);
			}
			for (int i = 1; i < numLowQubits; i += 2)
			{
				qc.cz(i, i - 1);
			}
		}
		else
			for (int i = 0; i < numQubits; i++)
			{
				if (i % 3 == 0)
					qc.x(i);
				if (i % 3 == 1)
					qc.y(i);
				if (i % 3 == 2)
					qc.z(i);
			}
		qc.barrier();
	}
	return qc;
}

int main()
{
	// Initialize the matrix dictionary
	Matrix<DTYPE>::initMatrixDict();

	// Initialize a quantum circuit
	QCircuit qc = separableqc();

	// Initialize the local state vector for each process
	// [NOTE] The initial state vector is not |00..0>
	Matrix<DTYPE> hostSv(lenSv, 1);

	// 模拟次数
	int numSimulations = 10;
	// 存储模拟时间的向量
	vector<double> simulationTimes;

	for (int times = 0; times < numSimulations; ++times)
	{
		for (int i = 0; i < numThreads; i++)
		{
			hostSv.data[i * ((1 << qc.numQubits) / numThreads) + i][0] = make_cuDoubleComplex(1.0 / sqrt(numThreads), 0);
		}

		// 获取开始时间
		auto start = chrono::high_resolution_clock::now();

		QuanPath(qc, hostSv, numThreads, numHighQubits, numLowQubits);

		// 获取结束时间
		auto end = chrono::high_resolution_clock::now();

		chrono::duration<double> duration = end - start;
		simulationTimes.push_back(duration.count());
		// cout << "Svsim simulation completed in " << duration.count() << " seconds." << endl;
		cout << "Simulation " << times + 1 << " completed in " << duration.count() << " seconds." << endl;
	}
	double average = accumulate(simulationTimes.begin(), simulationTimes.end(), 0.0) / simulationTimes.size();
	cout << "Average: " << average << std::endl;
	return 0;
}