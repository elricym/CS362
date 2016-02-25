package cs362;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by elric on 2/15/2016.
 */
public class LambdaMeansPredictor extends Predictor {
    double cluster_lambda;
    int clustering_training_iterations;
    List<Cluster> clusters;

    public LambdaMeansPredictor() {
        clusters = new ArrayList<>();

        clustering_training_iterations = 10;
        if (CommandLineUtilities.hasArg("clustering_training_iterations"))
            clustering_training_iterations = CommandLineUtilities.getOptionValueAsInt("clustering_training_iterations");
    }

    @Override
    public void train(List<Instance> instances) {
        FeatureVector meanVector = initializePrototype(instances);
        clusters.add(new Cluster(0, meanVector));
        cluster_lambda = initializeLambda(instances, meanVector);

        for (int i = 0; i < clustering_training_iterations; i++) {
            System.out.println("Iteration::" + i);
            E_step(instances);
            M_step();
        }
    }

    @Override
    public Label predict(Instance instance) {
        int assignment = 0;
        double minDistance = Double.MAX_VALUE;
        for (Cluster cluster : clusters) {
            double distance = computeNorm(instance.getFeatureVector(), cluster.getPrototype());
            if (distance < minDistance) {
                assignment = cluster.getId();
                minDistance = distance;
            }
        }
        return new ClassificationLabel(assignment);
    }

    private FeatureVector initializePrototype(List<Instance> instances) {
        FeatureVector meanVector = new FeatureVector();
        for (Instance instance : instances) {
            for (Map.Entry<Integer, Double> entry : instance.getFeatureVector().entrySet()) {
                if (meanVector.getKeySet().contains(entry.getKey())) {
                    meanVector.add(entry.getKey(), entry.getValue() / instances.size() + meanVector.get(entry.getKey()));
                } else {
                    meanVector.add(entry.getKey(), entry.getValue() / instances.size());
                }
            }
        }
        return meanVector;
    }

    private double initializeLambda(List<Instance> instances, FeatureVector meanVector) {
        double lambda = 0.0;
        if (CommandLineUtilities.hasArg("cluster_lambda"))
            lambda = CommandLineUtilities.getOptionValueAsFloat("cluster_lambda");
        else {
            for (Instance instance : instances) {
                lambda += computeNorm(instance.getFeatureVector(), meanVector);
            }
            lambda /= instances.size();
        }
        return lambda;
    }

    private double computeNorm(FeatureVector fv1, FeatureVector fv2) {
        double norm = 0.0;
        for (int index : fv1.getKeySet()) {
            if (fv2.getKeySet().contains(index)) {
                double distance = (fv1.get(index) - fv2.get(index));
                norm += distance * distance;
            } else {
                norm += fv1.get(index) * fv1.get(index);
            }
        }

        for (int index : fv2.getKeySet()) {
            if (!fv1.getKeySet().contains(index)) {
                norm += fv2.get(index) * fv2.get(index);
            }
        }
        return norm;
    }

    private void E_step(List<Instance> instances) {
        for (Instance instance : instances) {
            Cluster assignment = clusters.get(0);
            double minDistance = Double.MAX_VALUE;
            for (Cluster cluster : clusters) {
                double distance = computeNorm(instance.getFeatureVector(), cluster.getPrototype());
                if (distance < minDistance) {
                    assignment = cluster;
                    minDistance = distance;
                }
            }
            if (minDistance <= cluster_lambda) {
                assignment.getMembers().add(instance);
            } else {
                Cluster cluster = new Cluster(clusters.size(), instance.getFeatureVector());
                clusters.add(cluster);
                cluster.getMembers().add(instance);
            }
        }
    }

    private void M_step() {
        for (Cluster cluster : clusters) {
            FeatureVector meanVector = initializePrototype(cluster.getMembers());
            cluster.setPrototype(meanVector);
            cluster.getMembers().clear();
            System.gc();
        }
    }
}

class Cluster implements Serializable {
    private static final long serialVersionUID = 1L;
    private int id;
    private FeatureVector prototype;
    private List<Instance> members;

    public Cluster(int id, FeatureVector featureVector) {
        this.id = id;
        this.prototype = featureVector;
        members = new ArrayList<>();
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public FeatureVector getPrototype() {
        return prototype;
    }

    public void setPrototype(FeatureVector prototype) {
        this.prototype = prototype;
    }

    public List<Instance> getMembers() {
        return members;
    }

    public void setMembers(List<Instance> members) {
        this.members = members;
    }
}