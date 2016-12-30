package com.soarex.neural;

import net.objecthunter.exp4j.Expression;
import net.objecthunter.exp4j.ExpressionBuilder;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Soarex [Shumaf Lovpache] on 15.12.16.
 *
 * This code is part of the scientific work "The decision problem of regression analysis
 * using deep neural networks on the example of the approximation of mathematical functions"
 *
 * Example of program args: "sin(2*pi*x)" -10 10 1000
 * NOTE: The number of arguments must be strictly equal to 4!
 *
 * "sin(2*pi*x)"   - function to be approximated (it's desirable to be quoted)
 * -10             - minimum value at which we consider the function
 * 10              - maximum value at which we consider the function
 * 1000            - number of data points that will be used as training samples to the neural network
 *
 * number of output approximating functions = number of epochs (full passes of the data) / plotFrequency (How frequently should we plot the network output)
 */
public class RegressionMathFuncApproximation {

    public static String func;
    public static double xmin;
    public static double xmax;

    //Number of data points
    public static int nDataPoints;
    //How frequently should we plot the network output
    public static int plotFrequency = 500;
    //Random number generator seed, need for reproducibility
    public static int seed = 0;
    //Number of iterations per minibatch
    public static int iterations = 1;
    //Number of epochs (full passes of the data)
    public static int nEpochs = 2000;
    //Batch size: i.e., each epoch has nDataPoints/batchSize parameter updates
    public static int batchSize = 100;
    //Network learning rate
    public static double learningRate = 0.01;
    //Random number generator
    public static Random rng = new Random(seed);
    //Input (X) and output (Y). 1 because we calculate one-dimensional approximation (x, y). This algorithm can be adapted for N-dimension approximation.
    public static final int numInputs = 1;
    public static final int numOutputs = 1;

    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Incorrect args, using: java -jar rmt.jar \"function\" xMin xMax nDataPoints");
            System.exit(0);
        }

        func = args[0].toLowerCase();
        xmin = Double.parseDouble(args[1]);
        xmax = Double.parseDouble(args[2]);
        nDataPoints = Integer.parseInt(args[3]);

        //Generate the training data
        final INDArray x = Nd4j.linspace((int) xmin, (int) xmax, nDataPoints).reshape(nDataPoints, 1);
        final INDArray y = getFunctionValues(func, x);
        final DataSetIterator iterator = getTrainingData(x, y, batchSize, rng);

        //Create the network
        MultiLayerConfiguration config = getDeepNetworkConfiguration();
        final MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Train the network on the full data set, and evaluate in periodically
        final INDArray[] networkPredictions = new INDArray[nEpochs / plotFrequency];
        for (int i = 0; i < nEpochs; ++i) {
            iterator.reset();
            net.fit(iterator);
            if ((i + 1) % plotFrequency == 0) networkPredictions[i / plotFrequency] = net.output(x, false);
        }

        //Plot the target data and the network predictions
        plot(x, y, networkPredictions);
    }

    /**
     * Calculate function with given X points
     *
     * @param func - string representation of math function
     * @param x    - input X data
     * @return INDArray - an array of Y data
     */
    public static INDArray getFunctionValues(String func, INDArray x) {
        Expression expression = new ExpressionBuilder(func)
                .variables("x")
                .build();
        final double[] xd = x.data().asDouble();
        final double[] yd = new double[xd.length];
        for (int i = 0; i < xd.length; ++i) {
            yd[i] = expression.setVariable("x", xd[i]).evaluate();
        }
        return Nd4j.create(yd, new int[]{xd.length, 1});
    }

    /**
     * Returns the network configuration, 3 hidden DenseLayers of size 50.
     *
     * @return MultiLayerConfiguration - deep neural network model configuration
     */
    public static MultiLayerConfiguration getDeepNetworkConfiguration() {
        final int numHiddenNodes = 50;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.75)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(5, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(6, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(7, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }

    /**
     * Create a DataSetIterator for training
     *
     * @param x         X values
     * @param y         Y values of function
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng       Random number generator (for repeatability)
     */
    public static DataSetIterator getTrainingData(final INDArray x, final INDArray y, final int batchSize, final Random rng) {
        final List<DataSet> list = new DataSet(x, y).asList();
        Collections.shuffle(list, rng);
        return new ListDataSetIterator(list, batchSize);
    }

    /**
     * Plot the data
     *
     * @param x         - training data
     * @param y         - training data
     * @param predicted - neural network result
     */
    public static void plot(final INDArray x, final INDArray y, final INDArray... predicted) {
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, x, y, func);

        for (int i = 0; i < predicted.length; ++i) {
            addSeries(dataSet, x, predicted[i], String.valueOf(i));
        }

        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression example - f(x)=" + func,
                "x",
                "y",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    /**
     * This method used to simplify adding data to the chart
     *
     * @param dataSet - where the data will be added
     * @param x       - X data
     * @param y       - Y data
     * @param label   - label of the dataset
     */
    public static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label) {
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries xySeries = new XYSeries(label);
        for (int i = 0; i < xd.length; ++i) xySeries.add(xd[i], yd[i]);
        dataSet.addSeries(xySeries);
    }
}
