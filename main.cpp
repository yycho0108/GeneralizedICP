#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/LevenbergMarquardt>
/* #include <unsupported/Eigen/NonLinearOptimization> */
#include <vector>

#include <fmt/format.h>
#include <fmt/printf.h>

#include "nanoflann.hpp"

using Vector3f = Eigen::Vector3f;
using Transform = Eigen::Isometry3f;
using Cloud = Eigen::MatrixX3f;

Cloud TransformPointCloud(const Cloud& source_cloud,
    const Transform& transform)
{
    // TODO(yycho0108): implement
    Cloud out(source_cloud.rows(), source_cloud.cols());
    for (int i = 0; i < source_cloud.rows(); ++i) {
        out.row(i).transpose() = transform * source_cloud.row(i).transpose();
    }
    return out;
}

//template <typename T>
//struct TXTYTZRZParametrization {
//    static const std::size_t Dimensions = 4;
//    Eigen::Matrix<T, 3, 1> ApplyTransform(const Eigen::Matrix<T, 3, 1>& point);
//    void FromTransform(const Transform&);
//    Transform ToTransform();
//};

static constexpr const int kParamSize = 4;
template <typename T>
struct ParametrizedIsometryCost : public Eigen::DenseFunctor<T> {
    ParametrizedIsometryCost(const Cloud& source, const Cloud& target)
        : Eigen::DenseFunctor<T>(kParamSize, source.size())
        , source_(source)
        , target_(target)
    {
    }

    template <typename T1>
    int operator()(const Eigen::Matrix<T1, Eigen::Dynamic, 1>& parameter,
        Eigen::Matrix<T1, Eigen::Dynamic, 1>& fvec) const
    {
        const Eigen::Matrix<T1, 3, 1>& translation = parameter.template head<3>();
        //const Eigen::Matrix<T1, 3, 1>& rvec = parameter.template tail<3>();
        //const T1 angle = rvec.norm();
        //const Eigen::AngleAxis<T1>& rotation{angle, rvec / angle};
        //Eigen::Quaternion<T1> rotation;
        const Eigen::AngleAxis<T1>& rotation{ parameter(3), Eigen::Matrix<T1, 3, 1>{ 0, 0, 1 } };

        for (int i = 0; i < source_.rows(); ++i) {
            //fvec.segment(i * 3, 3) = source_.row(i).template cast<T1>() + translation.transpose() - target_.row(i).template cast<T1>();
            fvec.segment(i * 3, 3) = (rotation * source_.row(i).template cast<T1>().transpose()).transpose() + translation.transpose() - target_.row(i).template cast<T1>();
        }
        return 0;
        //// Set to simpler version for now.
        // Eigen::Transform<T1, 3, Eigen::Isometry> transform;
        // transform.translation() = translation;

        //// Will this work?
        // std::cout << "here1" << std::endl;
        // auto transformed = (transform.linear() * source_.transpose().template
        // cast<T1>()).transpose();
        ///* transformed.rowwise() += transform.translation().transpose(); */
        // std::cout << "here2" << std::endl;
        // Eigen::Matrix<T1, Eigen::Dynamic, 3> delta = (transformed -
        // target_).template cast<T1>(); std::cout << "here2.5" << std::endl; if
        // (!(delta.Flags & Eigen::RowMajorBit)) {
        //    // 3xN memory layout -> Nx3 memory layout
        //    Eigen::Matrix<T1, 3, Eigen::Dynamic> data = delta.transpose();
        //    Eigen::Map<Eigen::Matrix<T1, Eigen::Dynamic, 1>> rhs(data.data(),
        //        fvec.rows(), 1);
        //    fvec.segment(0, fvec.rows()) = rhs;
        //} else {
        //    Eigen::Map<Eigen::Matrix<T1, Eigen::Dynamic, 1>> rhs(delta.data(),
        //        fvec.rows(), 1);
        //    fvec.segment(0, fvec.rows()) = rhs;
        //}
        ////std::cout << "here3" << std::endl;
        ////std::cout << delta.rows() << 'x' << delta.cols() << std::endl;
        ////std::cout << fvec.rows() << 'x' << fvec.cols() << std::endl;
        ////std::cout << rhs.rows() << 'x' << rhs.cols() << std::endl;
        ////fvec = rhs;
        // return 0;
    };

    int df(const Eigen::Matrix<float, kParamSize, 1>& parameter,
        Eigen::MatrixXf& jac) const
    {
        using Scalar = Eigen::AutoDiffScalar<Eigen::VectorXf>;
        using ScalarVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        ScalarVector ax = parameter.template cast<Scalar>();
        ScalarVector av(this->values());

        for (int j = 0; j < this->values(); ++j) {
            av[j].derivatives().resize(this->inputs());
        }
        for (int i = 0; i < this->inputs(); ++i) {
            ax[i].derivatives().resize(this->inputs());
            ax[i].derivatives().setZero();
            ax[i].derivatives()(i) = 1.0;
            //ax[i].derivatives() = Eigen::Vector3f::Unit(this->inputs(), i);
            //ax[i].derivatives().resize(this->inputs());
        }

        operator()(ax, av);

        // jac = (48x3)
        for (int i = 0; i < this->values(); ++i) {
            // std::cout << "==" << std::endl;
            // std::cout << jac.row(i).rows() << std::endl;           // 1
            // std::cout << jac.row(i).cols() << std::endl;           // 3
            // std::cout << av[i].derivatives().rows() << std::endl;  // 0
            // std::cout << av[i].derivatives().cols() << std::endl;  // 1
            jac.row(i) = av[i].derivatives();
        }
    }
    const Cloud& source_;
    const Cloud& target_;
    /* IsometryCost<T> cost; */
};

/**
 * Generalized ICP implementation, based on
 * "Generalized-ICP" Segal et al.
 * www.roboticsproceedings.org/rss05/p21.pdf
 */
class GeneralizedICP {
public:
    using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<Cloud>;

    GeneralizedICP()
        : source(nullptr)
        , target(nullptr)
    {
    }
    void SetSourceCloud(const Cloud& source_cloud)
    {
        source = &source_cloud;
        /* source_tree = std::make_shared<KDTree>(3, std::cref(source_cloud),
         * 10); */
        /* source_tree->index->buildIndex(); */
    }
    void SetTargetCloud(const Cloud& target_cloud)
    {
        target = &target_cloud;
        target_tree = std::make_shared<KDTree>(3, std::cref(target_cloud), 10);
        target_tree->index->buildIndex();
    }

    void FindCorrespondences(const Cloud& source,
        std::vector<std::size_t>* indices,
        std::vector<float>* squared_distances)
    {
        // Reset memory.
        indices->clear();
        squared_distances->clear();
        indices->reserve(source.rows());
        squared_distances->reserve(source.rows());

        // Temps
        std::vector<long> knn_indices(5);
        std::vector<float> knn_squared_distances(5);

        for (int i = 0; i < source.rows(); ++i) {
            const Vector3f& query_point = source.row(i);
            /* fmt::print("query {}\n", query_point.transpose()); */
            /* const float* check = reinterpret_cast<const float*>(query_point.data()); */
            /* fmt::print("check {} {} {}\n", check[0], check[1], check[2]); */
            bool suc = target_tree->index->knnSearch(reinterpret_cast<const float*>(query_point.data()), 1,
                &knn_indices[0], &knn_squared_distances[0]);
            if (!suc) {
                fmt::print("unsuccessful");
                exit(1);
            }
            /* fmt::print("{},", knn_indices[0]); */

            indices->emplace_back(knn_indices[0]);
            squared_distances->emplace_back(knn_squared_distances[0]);
        }
    }

    float TransformCost(const Cloud& source, const Cloud& target,
        const Transform& transform)
    {
        // Nx3 = (3x3?) * (3xN)
        const Cloud& target_ = (transform * source.transpose()).transpose();
        const Cloud& delta = target - target_;

        /* delta = b - Ta */
        const Eigen::Matrix3f& R = transform.rotation();

        float total_cost{ 0 };
        for (int i = 0; i < source.rows(); ++i) {
            const Vector3f& d = delta.row(i);
            const float cost = d.transpose() * R * Eigen::Matrix3f::Identity() * R.transpose() * d;
            total_cost += cost;
        }
        return total_cost;
    }

    Transform OptimizeTransform(const Cloud& a, const Cloud& b, const Transform& seed = Transform::Identity())
    {
        // Initialize system.
        ParametrizedIsometryCost<float> cost_fun{ a, b };
        Eigen::LevenbergMarquardt<ParametrizedIsometryCost<float>> lm{ cost_fun };

        // Apply parametrization.
        Eigen::Matrix<float, -1, 1> param(4, 1);
        param.head<3>() = seed.translation();
        param(3) = Eigen::AngleAxis<float>{ seed.rotation() }.angle();

        // Run Optimization.
        lm.minimize(param);

        // Extract output and return.
        Transform out = Transform::Identity();
        out.translation() = param.head<3>();
        out.linear() = Eigen::AngleAxis<float>{ param(3), Eigen::Vector3f{ 0, 0, 1 } }.toRotationMatrix();
        return out;
    }

    Transform ComputeTransform()
    {
        static constexpr const float kTransformEpsilon = 0.001;
        static constexpr const float kRotationEpsilon = 0.001;
        static constexpr const float kMaxIterations = 100;

        Cloud tmp = *source;
        Transform transform = Transform::Identity();
        std::vector<std::size_t> target_indices;
        std::vector<float> squared_distances;

        bool converged = false;
        for (int count = 0; count < kMaxIterations; ++count) {
            FindCorrespondences(tmp, &target_indices, &squared_distances);
            for (std::size_t i = 0; i < source->rows(); ++i) {
                const std::size_t& source_index = i;
                const std::size_t& target_index = target_indices[i];
                const float& squared_distance = squared_distances[i];
                //fmt::print("{} {} {}\n", source_index, target_index,
                //    squared_distance);
            }

            // Make a copy for now. FIXME(yycho0108): better indexing scheme
            // const Cloud& X = target(target_indices, Eigen::internal::all);
            Cloud tmp_target(tmp.rows(), tmp.cols());
            for (std::size_t i = 0; i < source->rows(); ++i) {
                tmp_target.row(i) = target->row(target_indices[i]);
            }

            Transform target_from_tmp = OptimizeTransform(tmp, tmp_target);
            if (target_from_tmp.translation().squaredNorm() < kTransformEpsilon
                && std::abs(std::acos(0.5 * (target_from_tmp.linear().trace() - 1.0))) < kRotationEpsilon) {
                converged = true;
                break;
            }
            tmp = TransformPointCloud(tmp, target_from_tmp);
            transform = target_from_tmp * transform;

            ++count;
        }
        fmt::print("Converged : {}\n", bool(converged));
        return transform;
    }

    const Cloud *source, *target;
    /* std::shared_ptr<KDTree> source_tree; */
    std::shared_ptr<KDTree> target_tree;
};

int main()
{
    Transform ground_truth_transform = Transform::Identity();
    ground_truth_transform.translation() = Eigen::Vector3f{ 0.2, 0.2, 0.3 };
    ground_truth_transform.linear() = Eigen::AngleAxisf(0.1, Eigen::Vector3f{ 0., 0., 1. }).toRotationMatrix();

    Cloud source_cloud = Cloud::Zero(16, 3);
    source_cloud.setRandom();

    Cloud target_cloud = Cloud::Zero(17, 3);
    //target_cloud.block<16, 3>(0, 0) = source_cloud;
    target_cloud.block<16, 3>(0, 0) = TransformPointCloud(source_cloud, ground_truth_transform);

    /* fmt::print("{} vs {}", source_cloud.row(0), target_cloud.row(0)); */

    GeneralizedICP gicp;
    gicp.SetSourceCloud(source_cloud);
    gicp.SetTargetCloud(target_cloud);
    const Transform& estimated_transform = gicp.ComputeTransform();

    fmt::print("ground truth : \n{}\n", ground_truth_transform.matrix());
    fmt::print("computed : \n{}\n", estimated_transform.matrix());
    return 0;
}
