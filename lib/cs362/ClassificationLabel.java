package cs362;

import java.io.Serializable;

public class ClassificationLabel extends Label implements Serializable {

    private int label_value;

    public ClassificationLabel(int label) {
        // TODO Auto-generated constructor stub
        // Initialize label value with given label.
        this.label_value = label;
    }

    @Override
    public String toString() {
        // TODO Auto-generated method stub
        // Convert integer label into string.
        return label_value + "";
    }

    @Override
    public double getValue() {
        return label_value;
    }
}
