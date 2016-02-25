package cs362;

import cs362.nonlinear.kernel.GaussianKernel;
import cs362.nonlinear.kernel.LinearKernel;
import cs362.nonlinear.kernel.PolynomialKernel;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

import java.io.*;
import java.util.LinkedList;
import java.util.List;

public class Learn {
    static public LinkedList<Option> options = new LinkedList<Option>();

    public static void main(String[] args) throws IOException {
        // Parse the command line.
        String[] manditory_args = {"mode"};
        createCommandLineOptions();
        CommandLineUtilities.initCommandLineParameters(args, Learn.options, manditory_args);

        String mode = CommandLineUtilities.getOptionValue("mode");
        String data = CommandLineUtilities.getOptionValue("data");
        String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
        String algorithm = CommandLineUtilities.getOptionValue("algorithm");
        String model_file = CommandLineUtilities.getOptionValue("model_file");
        String task = CommandLineUtilities.getOptionValue("task"); // classification vs. regression

        boolean classify = true;

        if (task != null && task.equals("regression")) {
            classify = false;
        }

        if (mode.equalsIgnoreCase("train")) {
            if (data == null || algorithm == null || model_file == null) {
                System.out.println("Train requires the following arguments: data, algorithm, model_file");
                System.exit(0);
            }
            // Load the training data.
            DataReader data_reader = new DataReader(data, classify);
            List<Instance> instances = data_reader.readData();
            data_reader.close();

            // Train the model.
            Predictor predictor = train(instances, algorithm);
            saveObject(predictor, model_file);

        } else if (mode.equalsIgnoreCase("test")) {
            if (data == null || predictions_file == null || model_file == null) {
                System.out.println("Train requires the following arguments: data, predictions_file, model_file");
                System.exit(0);
            }

            // Load the test data.
            DataReader data_reader = new DataReader(data, classify);
            List<Instance> instances = data_reader.readData();
            data_reader.close();

            // Load the model.
            Predictor predictor = (Predictor) loadObject(model_file);
            evaluateAndSavePredictions(predictor, instances, predictions_file);
        } else {
            System.out.println("Requires mode argument.");
        }
    }


    private static Predictor train(
            List<Instance> instances, String algorithm) {
        // TODO Train the model using "algorithm" on "data"
        Predictor predictor = null;
        // Initialize predictor with selected algorithm.
        if (algorithm.equals("majority")) {
            predictor = new MajorityClassifier();
        } else if (algorithm.equals("even_odd")) {
            predictor = new EvenOddClassifier();
        } else if (algorithm.equals("linear_regression")) {
            predictor = new LinearRegression();
        } else if (algorithm.equals("naive_bayes")) {
            predictor = new NaiveBayesClassifier();
        } else if (algorithm.equals("perceptron")) {
            predictor = new PerceptronClassifier();
        } else if (algorithm.equals("kernel_logistic_regression")) {
            // Get commandline option "kernel".
            String kernel = CommandLineUtilities.getOptionValue("kernel");
            if (kernel == null) {
                throw new RuntimeException();
            } else if (kernel.equals("linear_kernel")) {
                predictor = new LinearKernel();
            } else if (kernel.equals("polynomial_kernel")) {
                predictor = new PolynomialKernel();
            } else if (kernel.equals("gaussian_kernel")) {
                predictor = new GaussianKernel();
            }
        } else if (algorithm.equals("lambda_means")) {
            predictor = new LambdaMeansPredictor();
        } else {
            throw new RuntimeException();
        }
        // Train the model by invoking train method.
        predictor.train(instances);
        // TODO Evaluate the model
        // Output the result of evaluation.
        System.out.println(new AccuracyEvaluator().evaluate(instances, predictor));
        return predictor;
    }

    private static void evaluateAndSavePredictions(Predictor predictor,
                                                   List<Instance> instances, String predictions_file) throws IOException {
        PredictionsWriter writer = new PredictionsWriter(predictions_file);
        // TODO Evaluate the model if labels are available.
        // If data has given labels, evaluate the model.
        if (instances.get(0).getLabel() != null)
            System.out.println("The accuracy is " + new AccuracyEvaluator().evaluate(instances, predictor));

        for (Instance instance : instances) {
            Label label = predictor.predict(instance);
            writer.writePrediction(label);
        }

        writer.close();

    }

    public static void saveObject(Object object, String file_name) {
        try {
            ObjectOutputStream oos =
                    new ObjectOutputStream(new BufferedOutputStream(
                            new FileOutputStream(new File(file_name))));
            oos.writeObject(object);
            oos.close();
        } catch (IOException e) {
            System.err.println("Exception writing file " + file_name + ": " + e);
        }
    }

    /**
     * Load a single object from a filename.
     *
     * @param file_name
     * @return
     */
    public static Object loadObject(String file_name) {
        ObjectInputStream ois;
        try {
            ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
            Object object = ois.readObject();
            ois.close();
            return object;
        } catch (IOException e) {
            System.err.println("Error loading: " + file_name);
        } catch (ClassNotFoundException e) {
            System.err.println("Error loading: " + file_name);
        }
        return null;
    }

    public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
        OptionBuilder.withArgName(arg_name);
        OptionBuilder.hasArg(has_arg);
        OptionBuilder.withDescription(description);
        Option option = OptionBuilder.create(option_name);

        Learn.options.add(option);
    }

    private static void createCommandLineOptions() {
        registerOption("data", "String", true, "The data to use.");
        registerOption("mode", "String", true, "Operating mode: train or test.");
        registerOption("predictions_file", "String", true, "The predictions file to create.");
        registerOption("algorithm", "String", true, "The name of the algorithm for training.");
        registerOption("model_file", "String", true, "The name of the model file to create/load.");
        registerOption("task", "String", true, "The name of the task (classification or regression).");

        // Other options will be added here.
        registerOption("lambda", "double", true, "The level of smoothing for Naive Bayes.");
        registerOption("online_learning_rate", "double", true, "The LTU learning rate.");
        registerOption("kernel", "String", true, "The kernel for Kernel Logistic regression [linear_kernel, polynomial_kernel, gaussian_kernel].");
        registerOption("polynomial_kernel_exponent", "double", true, "The exponent of the polynomial kernel.");
        registerOption("gaussian_kernel_sigma", "double", true, "The sigma of the Gaussian kernel.");
        registerOption("gradient_ascent_learning_rate", "double", true, "The learning rate for logistic regression.");
        registerOption("gradient_ascent_training_iterations", "int", true, "The number of training iterations.");
        registerOption("cluster_lambda", "double", true, "The value of lambda in lambda-means.");
        registerOption("clustering_training_iterations", "int", true, "The number of lambda-means EM iterations.");
    }
}
