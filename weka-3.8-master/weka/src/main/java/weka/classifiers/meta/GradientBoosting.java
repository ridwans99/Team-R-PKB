package weka.classifiers.meta;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.rules.ZeroR;

import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.UnassignedClassException;

public class GradientBoosting extends IteratedSingleClassifierEnhancer implements OptionHandler, 
        AdditionalMeasureProducer, WeightedInstancesHandler, TechnicalInformationHandler, IterativeClassifier {

  /** Untuk Serialisasi */
  static final long serialVersionUID = 1L;

  /** ArrayList untuk menyimpan pengklasifikasi dasar yang dihasilkan*/
  protected ArrayList<Classifier> m_Classifiers;
  
  /** Shrinkage (Learning rate). Default = no shrinkage. */
  protected double m_shrinkage = 0.005;

  /** Mean atau median */
  protected double m_InitialPrediction;

  /** apakah kita memiliki data yang sesuai atau tidak (jika hanya mean / mode yang digunakan) */
  protected boolean m_SuitableData = true;

  /** Data */
  protected Instances m_Data;

  /** Jumlah residu (absolut atau kuadrat).*/
  protected double m_Error;

  /** Peningkatan jumlah sisa (absolut atau kuadrat). */
  protected double m_Diff;
  
  /**  Meminimalkan kesalahan absolut daripada kesalahan kuadrat. */
  protected boolean m_MinimizeAbsoluteError;

  /**
   * Mengizinkan pelatihan untuk dilanjutkan setelah inisiasi
   * membangun Model.
   */
  protected boolean m_resume;

  /** Jumlah pengulangan yang dilakukan dalam sesi pengulangan ini */
  protected int m_numItsPerformed;
  
  /**
   * Mengembalikan string yang menjelaskan penilai atribut ini
   */
  public String globalInfo() {
    return " Meta classifier that enhances the performance of a regression "
      +"base classifier. Each iteration fits a model to the residuals left "
      +"by the classifier on the previous iteration. Prediction is "
      +"accomplished by adding the predictions of each classifier. "
      +"Reducing the shrinkage (learning rate) parameter helps prevent "
      +"overfitting and has a smoothing effect but increases the learning "
      +"time.\n\n"
      +"For more information see:\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.TECHREPORT);
    result.setValue(Field.AUTHOR, "J.H. Friedman");
    result.setValue(Field.YEAR, "2002");
    result.setValue(Field.TITLE, "Stochastic Gradient Boosting");
    result.setValue(Field.INSTITUTION, "Stanford University");
    result.setValue(Field.PS, "https://www.sciencedirect.com/science/article/abs/pii/S0167947301000652");
    
    return result;
  }

  /**
   * Default constructor menentukan DecisionStump sebagai klasifikasi
   */
  public GradientBoosting() {

    this(new weka.classifiers.trees.DecisionStump());
  }

  /**
   * Mengambil pengklasifikasi dasar sebagai argumen.
   *
   * @param classifier the base classifier to use
   */
  public GradientBoosting(Classifier classifier) {

    m_Classifier = classifier;
  }

  /**
   * String menjelaskan default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.DecisionStump";
  }

  /**
   * Mengembalikan enumerasi.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(2);

    newVector.addElement(new Option(
            "\tSpecify shrinkage rate. (default = 0.005, i.e., no shrinkage)",
            "S", 1, "-S"));

    newVector.addElement(new Option(
            "\tMinimize absolute error instead of squared error (assumes that base learner minimizes absolute error).",
            "A", 0, "-A"));

    newVector.addElement(new Option("\t" + resumeTipText() + "\n",
      "resume", 0, "-resume"));


    newVector.addAll(Collections.list(super.listOptions()));
    
    return newVector.elements();
  }

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -S
   *  Specify shrinkage rate. (default = 0.005, ie. no shrinkage)
   * </pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   *
   * <pre> -A
   *  Minimize absolute error instead of squared error (assumes that base learner minimizes absolute error).
   *  
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.DecisionStump)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.DecisionStump:
   * </pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String optionString = Utils.getOption('S', options);
    if (optionString.length() != 0) {
      Double temp = Double.valueOf(optionString);
      setShrinkage(temp.doubleValue());
    }
    setMinimizeAbsoluteError(Utils.getFlag('A', options));

    setResume(Utils.getFlag("resume", options));

    super.setOptions(options);
  }

  /**
   * Mendapatkan Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {
    
    Vector<String> options = new Vector<String>();

    options.add("-S"); options.add("" + getShrinkage());

    if (getMinimizeAbsoluteError()) {
      options.add("-A");
    }

    if (getResume()) {
      options.add("-resume");
    }

    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
  }

  /**
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String shrinkageTipText() {
    return "Shrinkage rate. Smaller values help prevent overfitting and "
            + "have a smoothing effect (but increase learning time). "
            +"Default = 0.005, ie. no shrinkage.";
  }

  /**
   * Atur parameter shrinkage
   *
   * @param l the shrinkage rate.
   */
  public void setShrinkage(double l) {
    m_shrinkage = l;
  }

  /**
   * Dapatkan nilai shrinkage.
   *
   * @return the value of the learning rate
   */
  public double getShrinkage() {
    return m_shrinkage;
  }

  /**
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minimizeAbsoluteErrorTipText() {
    return "Minimize absolute error instead of squared error (assume base learner minimizes absolute error)";
  }

  /**
   * Menyetel apakah kesalahan absolut akan diminimalkan.
   *
   * @param f true if absolute error is to be minimized.
   */
  public void setMinimizeAbsoluteError(boolean f) {
    m_MinimizeAbsoluteError = f;
  }

  /**
   * Dapatkan kesalahan absolut yang diminimalkan.
   *
   * @return true if absolute error is to be minimized
   */
  public boolean getMinimizeAbsoluteError() {
    return m_MinimizeAbsoluteError;
  }

  /**
   * @return the tool tip text for the finalize property
   */
  public String resumeTipText() {
    return "Set whether classifier can continue training after performing the"
      + "requested number of iterations. \n\tNote that setting this to true will "
      + "retain certain data structures which can increase the \n\t"
      + "size of the model.";
  }

  /**
   * @param resume true if the model is to be finalized after performing iterations
   */
  public void setResume(boolean resume) {
    m_resume = resume;
  }

  /**
   * Mengembalikan nilai model yang akan diselesaikan (atau telah diselesaikan) setelah
   * pelatihan.
   *
   * @return the current value of finalize
   */
  public boolean getResume() {
    return m_resume;
  }

  /**
   * Mengembalikan kemampuan default pengklasifikasi.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    
    return result;
  }

  /**
   * Metode yang digunakan untuk membangun pengklasifikasi.
   */
  public void buildClassifier(Instances data) throws Exception {

    // Initialize classifier
    initializeClassifier(data);

    // For the given number of iterations
    while (next()) {};

    // Clean up
    done();
  }

  /**
   * Inisialisasi pengklasifikasi.
   *
   * @param data the training data
   * @throws Exception if the classifier could not be initialized successfully
   */
  @Override public void initializeClassifier(Instances data) throws Exception {

    m_numItsPerformed = 0;

    if (m_Data == null) {
      // can classifier handle the data?
      getCapabilities().testWithFail(data);

      // remove instances with missing class
      m_Data = new Instances(data);
      m_Data.deleteWithMissingClass();

      // Add the model for the mean first
      if (getMinimizeAbsoluteError()) {
        m_InitialPrediction = m_Data
          .kthSmallestValue(m_Data.classIndex(), m_Data.numInstances() / 2);
      } else {
        m_InitialPrediction = m_Data.meanOrMode(m_Data.classIndex());
      }

      // only class? -> use only ZeroR model
      if (m_Data.numAttributes() == 1) {
        System.err.println(
          "Cannot build non-trivial model (only class attribute present in data!).");
        m_SuitableData = false;
        return;
      } else {
        m_SuitableData = true;
      }

      // Initialize list of classifiers and data
      m_Classifiers = new ArrayList<Classifier>(m_NumIterations);
      m_Data = residualReplace(m_Data, m_InitialPrediction);

      // Calculate error
      m_Error = 0;
      m_Diff = Double.MAX_VALUE;
      for (int i = 0; i < m_Data.numInstances(); i++) {
        if (getMinimizeAbsoluteError()) {
          m_Error += m_Data.instance(i).weight() * Math.abs(m_Data.instance(i).classValue());
        } else {
          m_Error +=
            m_Data.instance(i).weight() * m_Data.instance(i).classValue() * m_Data.instance(i).classValue();
        }
      }
      if (m_Debug) {
        if (getMinimizeAbsoluteError()) {
          System.err.println(
            "Sum of absolute residuals (predicting the median) : " + m_Error);
        } else {
          System.err.println(
            "Sum of squared residuals (predicting the mean) : " + m_Error);
        }
      }
    }
  }

  /**
   * Lakukan iterasi lain.
   */
  @Override public boolean next() throws Exception {

    if ((!m_SuitableData) || (m_numItsPerformed >= m_NumIterations) ||
            (m_Diff <= Utils.SMALL)) {
      return false;
    }

    // Build the classifier
    m_Classifiers.add(AbstractClassifier.makeCopy(m_Classifier));
    m_Classifiers.get(m_Classifiers.size() - 1).buildClassifier(m_Data);

    m_Data = residualReplace(m_Data, m_Classifiers.get(m_Classifiers.size() - 1));
    double sum = 0;
    for (int i = 0; i < m_Data.numInstances(); i++) {
      if (getMinimizeAbsoluteError()) {
        sum += m_Data.instance(i).weight() * Math.abs(m_Data.instance(i).classValue());
      } else {
        sum += m_Data.instance(i).weight() * m_Data.instance(i).classValue() * m_Data.instance(i).classValue();
      }
    }
    if (m_Debug) {
      if (getMinimizeAbsoluteError()) {
        System.err.println("Sum of absolute residuals: " + sum);
      } else {
        System.err.println("Sum of squared residuals: " + sum);
      }
    }
  
    m_Diff = m_Error - sum;
    m_Error = sum;
    m_numItsPerformed++;

    return true;
  }

  /**
   * Clean up.
   */
  @Override public void done() {
    if (!getResume()) {
      m_Data = null;
    }
  }

  /**
   * Klasifikasikan sebuah instance.
   *
   * @param inst the instance to predict
   * @return a prediction for the instance
   * @throws Exception if an error occurs
   */
  public double classifyInstance(Instance inst) throws Exception {

    double prediction = m_InitialPrediction;

    // default model?
    if (!m_SuitableData) {
      return prediction;
    }
    
    for (Classifier classifier : m_Classifiers) {
      double toAdd = classifier.classifyInstance(inst);
      if (Utils.isMissingValue(toAdd)) {
        throw new UnassignedClassException("GradientBoosting: base learner predicted missing value.");
      }
      prediction += (toAdd * getShrinkage());
    }

    return prediction;
  }

  /**
   * Ganti nilai kelas instance dari iterasi saat ini
   * dengan residu setelah memprediksi dengan pengklasifikasi yang disediakan.
   *
   * @param data the instances to predict
   * @param c the classifier to use
   * @return a new set of instances with class values replaced by residuals
   * @throws Exception if something goes wrong
   */
  private Instances residualReplace(Instances data, Classifier c) throws Exception {

    Instances newInst = new Instances(data);
    for (int i = 0; i < newInst.numInstances(); i++) {
      double pred = c.classifyInstance(newInst.instance(i));
      if (Utils.isMissingValue(pred)) {
        throw new UnassignedClassException("GradientBoosting: base learner predicted missing value.");
      }
      newInst.instance(i).setClassValue(newInst.instance(i).classValue() - (pred * getShrinkage()));
    }
    return newInst;
  }

  /**
   * Ganti nilai kelas instance dari iterasi saat ini
   * dengan residu setelah memprediksi konstanta yang diberikan.
   *
   * @param data the instances to predict
   * @param c the constant to use
   * @return a new set of instances with class values replaced by residuals
   * @throws Exception if something goes wrong
   */
  private Instances residualReplace(Instances data, double c) throws Exception {

    Instances newInst = new Instances(data);
    for (int i = 0; i < newInst.numInstances(); i++) {
      newInst.instance(i).setClassValue(newInst.instance(i).classValue() - c);
    }
    return newInst;
  }

  /**
   * Menampilkan enum
   * @return an enumeration of the measure names
   */
  public Enumeration<String> enumerateMeasures() {
    Vector<String> newVector = new Vector<String>(1);
    newVector.addElement("measureNumIterations");
    return newVector.elements();
  }

  /**
   * Menampilkan nilai value Measure
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.compareToIgnoreCase("measureNumIterations") == 0) {
      return measureNumIterations();
    } else {
      throw new IllegalArgumentException(additionalMeasureName 
			  + " not supported (GradientBoosting)");
    }
  }

  /**
   * menampilkan jumlah iterasi (pengklasifikasi dasar) selesai
   * @return the number of iterations (same as number of base classifier
   * models)
   */
  public double measureNumIterations() {
    return m_Classifiers.size();
  }

  /**
   * @return a description of the classifier as a string
   */
  public String toString() {
    StringBuffer text = new StringBuffer();
    
    if (m_SuitableData && m_Classifiers == null) {
      return "Classifier hasn't been built yet!";
    }

    // only ZeroR model?
    if (!m_SuitableData) {
      StringBuffer buf = new StringBuffer();
      buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
      buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
      buf.append("Warning: Non-trivial model could not be built, initial prediction is: ");
      buf.append(m_InitialPrediction);
      return buf.toString();
    }

    text.append("Gradient Boosting\n\n");

    text.append("Initial prediction: " + m_InitialPrediction + "\n\n");

    text.append("Base classifier " 
		+ getClassifier().getClass().getName()
		+ "\n\n");
    text.append("" + m_Classifiers.size() + " models generated.\n");

    for (int i = 0; i < m_Classifiers.size(); i++) {
      text.append("\nModel number " + i + "\n\n" +
		  m_Classifiers.get(i) + "\n");
    }

    return text.toString();
  }
  
  /**
   * Menampilkan string
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   * Metode utama untuk menguji kelas ini.
   *
   * @param argv should contain the following arguments:
   * -t training file [-T test file] [-c class index]
   */
  public static void main(String [] argv) {
    runClassifier(new GradientBoosting(), argv);
  }
}
