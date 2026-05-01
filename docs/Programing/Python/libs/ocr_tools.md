---
title: OCR_tools
---

### rapidocr_onnxruntime/rapidocr/onnxocr

- CPU性能：onnxocr（4s），rapid_ocr_onnxruntime（20s），rapidocr（24s）
- rapidocr可基于torch使用gpu

```python
from rapidocr_onnxruntime import RapidOCR as RapidOCR_Onnx
from rapidocr import RapidOCR, EngineType
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from pdf2image import convert_from_path
import numpy as np

imgs = convert_from_path(r"G:\VMware\shares/张小斌9.29收（寄信）.pdf", first_page=1, last_page=2)   # [first_page, last_page]，下标从1开始

def rapid_ocr_onnx(imgs):
    print("==="*10 + "rapid_ocr_onnx" + "==="*10)
    engine = RapidOCR_Onnx()
    
    result, _ = engine(imgs[0])
    print([item[1] for item in result])


def rapid_ocr(imgs):
    print("===" * 10 + "rapid_ocr" + "===" * 10)
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
            "EngineConfig.torch.use_cuda": False,   # True
            "EngineConfig.torch.gpu_id": 0,
        }
    )
    
    result = engine(imgs[0])
    print(result.txts)


def onnx_ocr(imgs):
    print("===" * 10 + "onnx_ocr" + "===" * 10)
    engine = ONNXPaddleOcr(use_gpu=False, lang="ch")
    
    _, result = engine(np.array(imgs[0]))
    print([item[0] for item in result])
```

