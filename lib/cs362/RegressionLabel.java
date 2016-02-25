package cs362;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {

    double lable_value;

    public RegressionLabel(double label) {
        // TODO Auto-generated constructor stub
        // Initialize label value with given label.
        this.lable_value = label;
    }

    @Override
    public String toString() {
        // TODO Auto-generated method stub
        // Convert integer label into string.
        return lable_value + "";
    }

    @Override
    public double getValue() {
        return lable_value;
    }

}
