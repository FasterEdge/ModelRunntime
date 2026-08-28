// Deep Java Library (DJL) 推理示例
// 动态参数均用注释占位，按实际模型修改

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;

public class Main {
    public static void main(String[] args) throws Exception {
        // ===== 动态参数（按实际模型修改）=====
        // MODEL_URL: 模型 URI（支持本地/远程，如 "djl://ai.djl.mxnet/resnet50"）
        String modelUrl = "file:///path/to/model";  // TODO: 替换为模型 URI
        // INPUT_SIZE: 输入图像尺寸
        int width = 224, height = 224;  // TODO: 按模型输入修改
        // NUM_CLASSES: 类别数
        int numClasses = 1000;  // TODO: 按模型实际类别数修改

        Criteria<Image, float[]> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, float[].class)
                .optModelUrls(modelUrl)
                .optTranslator(new BaseImageTranslator.Builder()
                        .addTransform(new Resize(width, height))
                        .addTransform(new ToTensor())
                        .build())
                .optProgress(new ProgressBar())
                .build();

        try (Model model = criteria.loadModel();
             Predictor<Image, float[]> predictor = model.newPredictor()) {
            // TODO: 替换输入图片
            Image img = ImageFactory.getInstance().fromUrl("file:///example.png");
            float[] out = predictor.predict(img);
            // TODO: 解析 out（分类为 argmax）
            System.out.println("inference done, output length: " + out.length);
        }
    }
}
