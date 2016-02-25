package cs362;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

/**
 * Created by elric on 1/27/2016.
 */
public class LinearRegression extends Predictor {

    private RealVector model;

    @Override
    public void train(List<Instance> instances) {
        double[][] trainingData = new double[instances.size()][];
        double[] labelSet = new double[instances.size()];
        for (int i = 0; i < instances.size(); i++) {
            FeatureVector fv = instances.get(i).getFeatureVector();
            for (int index : fv.getKeySet()) {
                trainingData[i][index] = fv.get(index);
            }
        }
        RealMatrix X = new Array2DRowRealMatrix(trainingData);
        RealVector Y = new ArrayRealVector(labelSet);
        // Compute w as model by using least square.
        model = (X.transpose().multiply(X)).power(-1).multiply(X.transpose()).operate(Y);
    }

    @Override
    public Label predict(Instance instance) {
        FeatureVector fv = instance.getFeatureVector();
        double[] data = new double[fv.N];
        for (int index : fv.getKeySet()) {
            data[index] = fv.get(index);
        }
        // Predict label by using the model.
        RealVector X = new ArrayRealVector(data);
        return new RegressionLabel(model.dotProduct(X));
    }
}
