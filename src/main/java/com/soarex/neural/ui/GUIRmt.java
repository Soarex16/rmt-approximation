package com.soarex.neural.ui;

import com.bulenkov.darcula.DarculaLaf;
import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import com.intellij.uiDesigner.core.Spacer;
import com.soarex.neural.util.CustomIterationListener;
import com.soarex.neural.util.MouseWheelHandler;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import static com.soarex.neural.RegressionMathFuncApproximation.*;

public class GUIRmt extends JFrame {
    private JPanel contentPane;
    private JButton buttonOK;
    private JTextField funcT;
    private JPanel chartJPanel;
    private JTextField fxMin;
    private JTextField fxMax;
    private JTextField fnDataPoints;
    private JProgressBar progressBar;
    private JPanel chartContainer;
    private JTextField layersCfg;
    private JTextField learningRateValue;
    private JButton buttonMoreLess;
    private JPanel optionsMore;
    private JTextField epochsValue;
    private JTextField numOutFunc;
    private JCheckBox randomSeedCheckBox;
    private JTextField seedValue;
    private JTextField batchSizeValue;
    private JSpinner iterationsValue;
    private JComboBox optimisationAlgo;
    private JComboBox lossFunc;
    private JButton buttonLearningGraph;
    private JLabel iterCL;
    private JButton buttonAbout;
    private JButton saveButton;
    private JButton loadButton;

    private OptimizationAlgorithm optAlg = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    private LossFunctions.LossFunction lossFn = LossFunctions.LossFunction.MSE;
    private int[] networkCfg;

    private ChartPanel chartPanel;

    private GUILearningProcess learn = new GUILearningProcess();
    private GUIAbout about = new GUIAbout();

    private int totalEpochs = 0;

    public GUIRmt() {
        try {
            UIManager.setLookAndFeel(new DarculaLaf());
        } catch (UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }
        $$$setupUI$$$();
        setContentPane(contentPane);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        getRootPane().setDefaultButton(buttonOK);
        try {
            setIconImage(ImageIO.read(getClass().getResourceAsStream("/64.png")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        iterationsValue.setValue(Integer.valueOf(1));
        optimisationAlgo.setSelectedIndex(4);
        lossFunc.setSelectedIndex(0);

        randomSeedCheckBox.addActionListener(e -> rngOff());
        buttonOK.addActionListener(e -> onOK());
        buttonMoreLess.addActionListener(e -> moreLess());
        buttonAbout.addActionListener(e -> credits());
        buttonLearningGraph.addActionListener(e -> learningGraph());
        saveButton.addActionListener(e -> saveLoadCfg(true));
        loadButton.addActionListener(e -> saveLoadCfg(false));
    }

    private void saveLoadCfg(boolean save) {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select JSON configuration file");
        chooser.showSaveDialog(this);
        String cfgFile = chooser.getSelectedFile().getAbsolutePath();

        if (save) {
            JSONObject obj = new JSONObject();
            obj.put("func", func);
            obj.put("xmin", xmin);
            obj.put("xmax", xmax);
            obj.put("nDataPoints", nDataPoints);
            obj.put("plotFrequency", plotFrequency);
            obj.put("seed", seed);
            obj.put("iterations", iterations);
            obj.put("nEpochs", nEpochs);
            obj.put("batchSize", batchSize);
            obj.put("learningRate", learningRate);
            obj.put("totalEpochs", totalEpochs);
            obj.put("numOutFunc", Integer.parseInt(numOutFunc.getText()));

            String[] nCfg = layersCfg.getText().replace(" ", "").split(",");
            networkCfg = new int[nCfg.length];
            for (int i = 0; i < nCfg.length; ++i) {
                networkCfg[i] = Integer.parseInt(nCfg[i]);
            }
            JSONArray layersArray = new JSONArray();
            for (int i = 0; i < networkCfg.length; ++i) {
                layersArray.add(i, networkCfg[i]);
            }
            obj.put("networkCfg", layersArray);

            obj.put("optAlg", optimisationAlgo.getSelectedIndex());
            obj.put("lossFn", lossFunc.getSelectedIndex());
            try (FileWriter file = new FileWriter(cfgFile)) {
                File f = new File(cfgFile);

                file.write(obj.toJSONString());
                file.flush();
                file.close();

                ChartUtilities.saveChartAsPNG(new File(f.getAbsolutePath() + "-approximation.png"), chartPanel.getChart(), 1920, 1080);
                ChartUtilities.saveChartAsPNG(new File(f.getAbsolutePath() + "-learning.png"), learn.chartPanel.getChart(), 1920, 1080);
            } catch (IOException e) {
                JOptionPane.showMessageDialog(null, "Error while saving configuration!", "Saving error", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        } else {
            try {
                JSONParser parser = new JSONParser();
                JSONObject obj = (JSONObject) parser.parse(cfgFile);
                funcT.setText((String) obj.get("func"));
                fxMin.setText(String.valueOf((double) obj.get("xmin")));
                fxMax.setText(String.valueOf((double) obj.get("xmax")));
                fnDataPoints.setText(String.valueOf((int) obj.get("nDataPoints")));
                seedValue.setText(String.valueOf((int) obj.get("seed")));
                iterationsValue.setValue(String.valueOf((int) obj.get("iterations")));
                epochsValue.setText(String.valueOf((int) obj.get("nEpochs")));
                batchSizeValue.setText(String.valueOf((int) obj.get("batchSize")));
                learningRateValue.setText(String.valueOf((double) obj.get("learningRate")));
                numOutFunc.setText(String.valueOf((int) obj.get("numOutFunc")));

                JSONArray layersArray = (JSONArray) obj.get("networkCfg");
                for (int i = 0; i < layersArray.size(); ++i) {
                    networkCfg[i] = (Integer) layersArray.get(i);
                }

                optAlg = OptimizationAlgorithm.values()[(int) obj.get("optAlg")];
                lossFn = LossFunctions.LossFunction.values()[(int) obj.get("lossFn")];
            } catch (ParseException e) {
                JOptionPane.showMessageDialog(null, "Error while loading configuration! Please, select another file.", "Configuration parsing error", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }

    private void credits() {
        if (about.isVisible()) {
            about.setVisible(false);
        } else {
            about.setVisible(true);
        }
    }

    private void learningGraph() {
        if (learn.isVisible()) {
            learn.setVisible(false);
        } else {
            learn.setVisible(true);
        }
    }

    private void rngOff() {
        if (randomSeedCheckBox.isSelected()) {
            seedValue.setEnabled(false);
        } else {
            seedValue.setEnabled(true);
        }
    }

    private void moreLess() {
        if (optionsMore.isVisible()) {
            optionsMore.setVisible(false);
            buttonMoreLess.setText("More...");
            this.pack();
        } else {
            optionsMore.setVisible(true);
            buttonMoreLess.setText("Less...");
            this.pack();
        }
    }

    private void onOK() {
        if (getThreadByName("soarex-neuro") != null) {
            JOptionPane.showMessageDialog(null, "Another learning process is already running", "Please, wait", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        if (randomSeedCheckBox.isSelected()) {
            seed = new Random().nextInt();
        } else {
            seed = Integer.parseInt(seedValue.getText());
        }
        optAlg = OptimizationAlgorithm.values()[optimisationAlgo.getSelectedIndex()];
        lossFn = LossFunctions.LossFunction.values()[lossFunc.getSelectedIndex()];
        func = funcT.getText().toLowerCase();
        xmin = Double.parseDouble(fxMin.getText());
        xmax = Double.parseDouble(fxMax.getText());
        nDataPoints = Integer.parseInt(fnDataPoints.getText());
        batchSize = Integer.parseInt(batchSizeValue.getText());
        nEpochs = Integer.parseInt(epochsValue.getText());
        iterations = (Integer) iterationsValue.getValue();
        learningRate = Double.parseDouble(learningRateValue.getText());
        plotFrequency = nEpochs / Integer.parseInt(numOutFunc.getText());

        int batches = nDataPoints / batchSize;
        if (batches < 1) batches = 1;
        totalEpochs = nEpochs * iterations * batches;

        String[] nCfg = layersCfg.getText().replace(" ", "").split(",");
        networkCfg = new int[nCfg.length];
        for (int i = 0; i < nCfg.length; ++i) {
            networkCfg[i] = Integer.parseInt(nCfg[i]);
        }

        final INDArray x = Nd4j.linspace((int) xmin, (int) xmax, nDataPoints).reshape(nDataPoints, 1);
        final INDArray y = getFunctionValues(func, x);
        final DataSetIterator iterator = getTrainingData(x, y, batchSize, rng);

        learn.clearChart();
        learn.setVisible(true);

        Thread networkThread = new Thread(() -> {
            plotFunc(x, y);
            progressBar.setMaximum(totalEpochs);
            progressBar.setValue(0);

            //Create the network
            MultiLayerConfiguration config = getParametrisedDeepNetworkConfiguration();
            final MultiLayerNetwork net = new MultiLayerNetwork(config);
            net.init();
            net.setListeners(new CustomIterationListener(1, totalEpochs, iterCL, progressBar, learn));

            //Train the network on the full data set, and evaluate in periodically
            final INDArray[] networkPredictions = new INDArray[nEpochs / plotFrequency];

            for (int i = 0; i < nEpochs; ++i) {
                iterator.reset();
                net.fit(iterator);
                if ((i + 1) % plotFrequency == 0) networkPredictions[i / plotFrequency] = net.output(x, false);
            }

            plotFunc(x, y, networkPredictions);
        });
        networkThread.setName("soarex-neuro");
        networkThread.start();
    }

    public Thread getThreadByName(String threadName) {
        for (Thread t : Thread.getAllStackTraces().keySet()) {
            if (t.getName().equals(threadName)) return t;
        }
        return null;
    }

    public MultiLayerConfiguration getParametrisedDeepNetworkConfiguration() {
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(optAlg)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.75)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(networkCfg[0])
                        .activation("tanh")
                        .build());
        /*if (networkCfg.length <= 1) {
            JOptionPane.showMessageDialog(null, "Configuration of  layers of the neural network is incorrect. Note that the number of parameters must be greater than 1.", "Incorrect configuration", JOptionPane.INFORMATION_MESSAGE);
        }*/ //bug
        for (int i = 0; i < networkCfg.length; ++i) {
            listBuilder.layer(i + 1, new DenseLayer.Builder().nIn(networkCfg[i]).nOut(i + 1 < networkCfg.length ? networkCfg[i + 1] : networkCfg[i])
                    .activation("tanh")
                    .build());
        }
        listBuilder.layer(networkCfg.length + 1, new OutputLayer.Builder(lossFn)
                .activation("identity")
                .nIn(networkCfg[networkCfg.length - 1]).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
        return listBuilder.build();
    }

    private void plotFunc(final INDArray x, final INDArray y, final INDArray... predicted) {
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, x, y, func);

        for (int i = 0; i < predicted.length; ++i) {
            addSeries(dataSet, x, predicted[i], String.valueOf(i));
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "x",
                "y",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(new Color(0xE8E8E8));
        chart.getXYPlot().setRangePannable(true);
        chart.getXYPlot().setDomainPannable(true);
        chartPanel.setChart(chart);
    }

    public static void main(String[] args) {
        GUIRmt form = new GUIRmt();
        form.setTitle("Math function approximation demo");
        form.pack();
        form.setVisible(true);
    }

    private void createUIComponents() {
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "x",
                "y",
                null,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(new Color(0xE8E8E8));
        chartPanel = new ChartPanel(chart);
        chartPanel.addMouseWheelListener(new MouseWheelHandler(chartPanel));
        chartPanel.setFillZoomRectangle(false);
        chartJPanel = new JPanel(new BorderLayout());
        chartJPanel.add(chartPanel);
        optimisationAlgo = new JComboBox(OptimizationAlgorithm.values());
        lossFunc = new JComboBox(LossFunctions.LossFunction.values());
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        createUIComponents();
        contentPane = new JPanel();
        contentPane.setLayout(new GridLayoutManager(4, 2, new Insets(10, 10, 10, 10), -1, -1));
        contentPane.setForeground(new Color(-14606047));
        contentPane.setToolTipText("");
        chartContainer = new JPanel();
        chartContainer.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        chartContainer.setForeground(new Color(-4473925));
        contentPane.add(chartContainer, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        chartContainer.add(chartJPanel, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        final JPanel panel1 = new JPanel();
        panel1.setLayout(new GridLayoutManager(2, 8, new Insets(0, 0, 0, 0), -1, -1));
        contentPane.add(panel1, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        progressBar = new JProgressBar();
        panel1.add(progressBar, new GridConstraints(0, 0, 1, 7, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label1 = new JLabel();
        label1.setText("Data points");
        panel1.add(label1, new GridConstraints(1, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label2 = new JLabel();
        label2.setText("Min");
        panel1.add(label2, new GridConstraints(1, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label3 = new JLabel();
        label3.setText("Max");
        panel1.add(label3, new GridConstraints(1, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        fnDataPoints = new JTextField();
        fnDataPoints.setText("");
        fnDataPoints.setToolTipText("Number of data points");
        fnDataPoints.setVisible(true);
        panel1.add(fnDataPoints, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        fxMin = new JTextField();
        fxMin.setText("");
        panel1.add(fxMin, new GridConstraints(1, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        fxMax = new JTextField();
        fxMax.setText("");
        panel1.add(fxMax, new GridConstraints(1, 6, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        iterCL = new JLabel();
        iterCL.setHorizontalAlignment(10);
        iterCL.setText("0/0");
        panel1.add(iterCL, new GridConstraints(0, 7, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel2 = new JPanel();
        panel2.setLayout(new GridLayoutManager(1, 3, new Insets(0, 0, 0, 0), -1, -1));
        panel2.setForeground(new Color(-2039584));
        contentPane.add(panel2, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, 1, null, null, null, 0, false));
        final JPanel panel3 = new JPanel();
        panel3.setLayout(new GridLayoutManager(1, 2, new Insets(0, 0, 0, 0), -1, -1));
        panel3.setForeground(new Color(-2039584));
        panel2.add(panel3, new GridConstraints(0, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        buttonOK = new JButton();
        buttonOK.setText("OK");
        panel3.add(buttonOK, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonLearningGraph = new JButton();
        buttonLearningGraph.setText("Learning");
        buttonLearningGraph.setToolTipText("Shows a graph of the learning process");
        panel3.add(buttonLearningGraph, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        funcT = new JTextField();
        funcT.setText("");
        funcT.setToolTipText("<html>\n<b>Predefined available functions:</b>\n<br>abs: absolute value\n<br>acos: arc cosine\n<br>asin: arc sine\n<br>atan: arc tangent\n<br>cbrt: cubic root\n<br>ceil: nearest upper integer\n<br>cos: cosine\n<br>cosh: hyperbolic cosine\n<br>exp: euler's number raised to the power (e^x)\n<br>floor: nearest lower integer\n<br>log: logarithmus naturalis (base e)\n<br>log10: logarithm (base 10)\n<br>log2: logarithm (base 2)\n<br>sin: sine\n<br>sinh: hyperbolic sine\n<br>sqrt: square root\n<br>tan: tangent\n<br>tanh: hyperbolic tangent\n<br>signum: signum function\n</html>");
        panel2.add(funcT, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        final JLabel label4 = new JLabel();
        label4.setText("f(x) =");
        label4.setToolTipText("Function to be approximated");
        panel2.add(label4, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel4 = new JPanel();
        panel4.setLayout(new GridLayoutManager(2, 5, new Insets(0, 0, 0, 0), -1, -1));
        panel4.setVisible(true);
        contentPane.add(panel4, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        optionsMore = new JPanel();
        optionsMore.setLayout(new GridLayoutManager(7, 7, new Insets(0, 0, 0, 0), -1, -1));
        optionsMore.setVisible(false);
        panel4.add(optionsMore, new GridConstraints(1, 0, 1, 5, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        layersCfg = new JTextField();
        layersCfg.setText("50, 50, 50, 50, 50, 50, 50, 50");
        layersCfg.setToolTipText("The number of neurons in the hidden layers (separated by commas).");
        layersCfg.setVisible(true);
        optionsMore.add(layersCfg, new GridConstraints(0, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label5 = new JLabel();
        label5.setText("Network configuration");
        optionsMore.add(label5, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        learningRateValue = new JTextField();
        learningRateValue.setText("0.01");
        learningRateValue.setToolTipText("Network learning rate");
        optionsMore.add(learningRateValue, new GridConstraints(1, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label6 = new JLabel();
        label6.setText("Learning rate");
        optionsMore.add(label6, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        epochsValue = new JTextField();
        epochsValue.setText("2000");
        epochsValue.setToolTipText("Number of epochs (full passes of the data)");
        optionsMore.add(epochsValue, new GridConstraints(2, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        numOutFunc = new JTextField();
        numOutFunc.setText("4");
        numOutFunc.setToolTipText("Number of output approximating functions");
        optionsMore.add(numOutFunc, new GridConstraints(3, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label7 = new JLabel();
        label7.setText("Epochs");
        optionsMore.add(label7, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label8 = new JLabel();
        label8.setText("Number of output functions");
        label8.setVisible(true);
        optionsMore.add(label8, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label9 = new JLabel();
        label9.setText("Seed");
        optionsMore.add(label9, new GridConstraints(4, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        randomSeedCheckBox = new JCheckBox();
        randomSeedCheckBox.setHideActionText(false);
        randomSeedCheckBox.setSelected(true);
        randomSeedCheckBox.setText("random seed");
        optionsMore.add(randomSeedCheckBox, new GridConstraints(4, 6, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label10 = new JLabel();
        label10.setText("Batch size");
        optionsMore.add(label10, new GridConstraints(4, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        batchSizeValue = new JTextField();
        batchSizeValue.setText("100");
        batchSizeValue.setToolTipText("Batch size: i.e., each epoch has nDataPoints/batchSize parameter updates");
        optionsMore.add(batchSizeValue, new GridConstraints(4, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label11 = new JLabel();
        label11.setText("Iterations");
        optionsMore.add(label11, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        iterationsValue = new JSpinner();
        iterationsValue.setToolTipText("Number of iterations per minibatch");
        optionsMore.add(iterationsValue, new GridConstraints(4, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label12 = new JLabel();
        label12.setText("Optimisation algorithm");
        optionsMore.add(label12, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        optimisationAlgo.setToolTipText("Optimization algorithm to use");
        optionsMore.add(optimisationAlgo, new GridConstraints(5, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label13 = new JLabel();
        label13.setText("Loss function");
        optionsMore.add(label13, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        lossFunc.setToolTipText("<html><b>Loss functions representing the price paid for inaccuracy of predictions</b><br>\n<b>MSE:</b> Mean Squared Error: Linear Regression<br>\n<b>EXPLL:</b> Exponential log likelihood: Poisson Regression<br>\n<b>XENT:</b> Cross Entropy: Binary Classification<br>\n<b>MCXENT:</b> Multiclass Cross Entropy<br>\n<b>RMSE_XENT:</b> RMSE Cross Entropy<br>\n<b>SQUARED_LOSS:</b> Squared Loss<br>\n<b>NEGATIVELOGLIKELIHOOD:</b> Negative Log Likelihood<br>\n</html>");
        optionsMore.add(lossFunc, new GridConstraints(6, 1, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        seedValue = new JTextField();
        seedValue.setEnabled(false);
        seedValue.setToolTipText("Random number generator seed, need for reproducibility");
        optionsMore.add(seedValue, new GridConstraints(4, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer1 = new Spacer();
        panel4.add(spacer1, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        buttonMoreLess = new JButton();
        buttonMoreLess.setText("More...");
        panel4.add(buttonMoreLess, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        saveButton = new JButton();
        saveButton.setText("Save");
        saveButton.setToolTipText("Save training graphs and approximation and the parameters of the neural network");
        panel4.add(saveButton, new GridConstraints(0, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonAbout = new JButton();
        buttonAbout.setText("About");
        panel4.add(buttonAbout, new GridConstraints(0, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        loadButton = new JButton();
        loadButton.setText("Load");
        loadButton.setToolTipText("Load the configuration of the neural network from file");
        panel4.add(loadButton, new GridConstraints(0, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer2 = new Spacer();
        contentPane.add(spacer2, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return contentPane;
    }
}
