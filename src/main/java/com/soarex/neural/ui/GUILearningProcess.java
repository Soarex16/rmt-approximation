package com.soarex.neural.ui;

import com.bulenkov.darcula.DarculaLaf;
import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import com.intellij.uiDesigner.core.Spacer;
import com.soarex.neural.util.MouseWheelHandler;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class GUILearningProcess extends JFrame {
    private JPanel contentPane;
    private JPanel chartContainer;
    private JPanel chartJPanel;
    public JLabel score;
    public JLabel iter;
    public ChartPanel chartPanel;

    public XYSeriesCollection learnData = new XYSeriesCollection();
    public XYSeries learnSeries = new XYSeries("Score/Iteration");

    public GUILearningProcess() {
        try {
            UIManager.setLookAndFeel(new DarculaLaf());
        } catch (UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }
        $$$setupUI$$$();
        setTitle("Learning process");
        setContentPane(contentPane);
        try {
            setIconImage(ImageIO.read(GUIRmt.class.getResourceAsStream("/64.png")));
        } catch (IOException e) {
            e.printStackTrace();
        }
        pack();
    }

    public void clearChart() {
        learnSeries = new XYSeries("Score/Iteration");
        learnData.removeAllSeries();
        learnData.addSeries(learnSeries);
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "iterations",
                "score",
                learnData,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(new Color(0xE8E8E8));
        chart.removeLegend();
        chart.getXYPlot().setRangePannable(true);
        chart.getXYPlot().setDomainPannable(true);
        chart.fireChartChanged();
        chartPanel.setChart(chart);
    }

    private void createUIComponents() {
        learnData.addSeries(learnSeries);
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "iterations",
                "score",
                learnData,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(new Color(0xE8E8E8));
        chart.removeLegend();
        chart.getXYPlot().setRangePannable(true);
        chart.getXYPlot().setDomainPannable(true);
        chart.fireChartChanged();
        chartPanel = new ChartPanel(chart);
        chartPanel.addMouseWheelListener(new MouseWheelHandler(chartPanel));
        chartPanel.setFillZoomRectangle(false);
        chartJPanel = new JPanel(new BorderLayout());
        chartJPanel.add(chartPanel);
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
        contentPane.setLayout(new GridLayoutManager(2, 5, new Insets(10, 10, 15, 10), -1, -1));
        chartContainer = new JPanel();
        chartContainer.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        contentPane.add(chartContainer, new GridConstraints(0, 0, 1, 5, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        chartJPanel.setAlignmentX(0.5f);
        chartContainer.add(chartJPanel, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        final JLabel label1 = new JLabel();
        label1.setAlignmentX(0.5f);
        label1.setText("Score:");
        contentPane.add(label1, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        score = new JLabel();
        score.setText("0");
        contentPane.add(score, new GridConstraints(1, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label2 = new JLabel();
        label2.setText("Iteration:");
        contentPane.add(label2, new GridConstraints(1, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        iter = new JLabel();
        iter.setText("0");
        contentPane.add(iter, new GridConstraints(1, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer1 = new Spacer();
        contentPane.add(spacer1, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return contentPane;
    }
}
