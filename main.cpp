#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // for calculating the execution time of program
#include <filesystem> // for reading files from directory

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

class FuzzyCMeans {
public:
    FuzzyCMeans(int clusters, int maxIter, float epsilon, float fuzziness)
        : numClusters(clusters), maxIterations(maxIter), epsilon(epsilon), m(fuzziness) {}

    void fit(const Mat& img) {
        int rows = img.rows;
        int cols = img.cols;

        // Reshape image into a 2D array of pixels
        Mat data = img.reshape(1, rows * cols);
        data.convertTo(data, CV_32F);

        // Initialize membership matrix with random values
        U = Mat::zeros(numClusters, data.rows, CV_32F);
        randu(U, Scalar(0), Scalar(1));

        // Normalize membership matrix
        normalize(U, U, 1, 0, NORM_L1, -1, Mat());

        for (int iter = 0; iter < maxIterations; ++iter) {
            // Compute cluster centers
            vector<Vec3f> clusterCenters(numClusters, Vec3f(0, 0, 0));
            for (int c = 0; c < numClusters; ++c) {
                float denominator = 0.0;
                for (int i = 0; i < data.rows; ++i) {
                    float um = pow(U.at<float>(c, i), m);
                    Vec3f pixel = data.at<Vec3f>(i, 0);
                    clusterCenters[c] += um * pixel;
                    denominator += um;
                }
                clusterCenters[c] /= denominator;
            }

            // Update membership matrix
            Mat U_new = Mat::zeros(U.size(), U.type());
            for (int i = 0; i < data.rows; ++i) {
                for (int c = 0; c < numClusters; ++c) {
                    float sum = 0.0;
                    for (int ck = 0; ck < numClusters; ++ck) {
                        float distRatio = norm(data.at<Vec3f>(i, 0) - clusterCenters[c]) /
                                          norm(data.at<Vec3f>(i, 0) - clusterCenters[ck]);
                        sum += pow(distRatio, 2 / (m - 1));
                    }
                    U_new.at<float>(c, i) = 1.0f / sum;
                }
            }

            // Check for convergence
            if (norm(U_new - U) < epsilon) {
                cout << "Converged after " << iter + 1 << " iterations." << endl;
                break;
            }

            U = U_new;
        }

        // Assign each pixel to the cluster with highest membership
        segmented = Mat::zeros(data.size(), CV_32FC1);
        for (int i = 0; i < data.rows; ++i) {
            float maxVal = 0.0;
            int maxIdx = 0;
            for (int c = 0; c < numClusters; ++c) {
                float val = U.at<float>(c, i);
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = c;
                }
            }
            segmented.at<float>(i, 0) = maxIdx;
        }

        segmented = segmented.reshape(0, rows);
    }

    Mat getSegmented() const {
        return segmented;
    }

private:
    int numClusters;
    int maxIterations;
    float epsilon;
    float m; // fuzziness parameter
    Mat U; // membership matrix
    Mat segmented; // segmented image
};

int main() {
    // Specify the input and output directories
    string inputDir = "C:/Users/admin/Desktop/IIT kanpur/Thesis/Image Segmentation/data/";
    string outputDir = "C:/Users/admin/Desktop/IIT kanpur/Thesis/Image Segmentation/segmented_output/";

    // Ensure the output directory exists
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // Start the timer to calculate program execution time
    auto startTime = std::chrono::system_clock::now();

    // Iterate through the files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        // Check if the file is an image (e.g., jpg, png, .jpeg)
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
            string imagePath = entry.path().string();
            cout << "Processing image: " << imagePath << endl;

            // Read the image
            Mat img = imread(imagePath, IMREAD_COLOR);

            // Check if the image is loaded successfully
            if (img.empty()) {
                cout << "Error: Could not open or find the image: " << imagePath << endl;
                continue;
            }

            // Convert image to floating point and normalize
            img.convertTo(img, CV_32F);
            img /= 255.0;

            // Apply Fuzzy C-Means algorithm
            FuzzyCMeans fcm(3, 1000, 1e-5, 2.0); // 3 clusters, 1000 max iterations, epsilon, fuzziness
            fcm.fit(img);

            // Get segmented image and convert back to 8-bit for display
            Mat segmented = fcm.getSegmented();
            segmented.convertTo(segmented, CV_8U, 255.0 / (3 - 1));

            // Construct output file name
            string outputPath = outputDir + entry.path().stem().string() + "_segmented.png";

            // Save the segmented image
            imwrite(outputPath, segmented);

            cout << "Segmented image saved to: " << outputPath << endl;
        }
    }

    // End the timer
    auto endTime = std::chrono::system_clock::now();

    // Calculate and display the duration of the program
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    cout << "Time taken by the program: " << duration << " seconds" << endl;

    return 0;
}
