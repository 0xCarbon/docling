import logging
from typing import Iterable

from docling_core.types.doc import BoundingBox, CoordOrigin

from docling.datamodel.base_models import OcrCell, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import LLMProviderOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

from PIL import Image
from io import BytesIO
import base64

_log = logging.getLogger(__name__)

class LLMProviderModel(BaseOcrModel):
    def __init__(self, enabled: bool, options: LLMProviderOptions):
        super().__init__(enabled=enabled, options=options)
        self.options: LLMProviderOptions

        if self.options.api_key is None:
            raise ValueError("api_key is required")

        if self.options.api_url is None:
            raise ValueError("api_url is required")
        
        self.scale = 3  # multiplier for 72 dpi == 216 dpi. TODO: check if this is necessary

        if self.enabled:
            install_errmsg = (
                "openai is not installed. Please install it via `pip install openai` to use this OCR engine. "
                "Alternatively, Docling has support for other OCR engines. See the documentation: "
                "https://ds4sd.github.io/docling/installation/"
            )
            connection_errmsg = (
                "Failed to connect to the API. Please check your API key and API URL. "
            )

            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(install_errmsg)
            try:
                self._client = OpenAI(api_key=self.options.api_key, base_url=self.options.api_url)
            except Exception as e:
                raise ConnectionError(connection_errmsg)
            
            if self.options.model is None:
                models = self._client.models.list()
                self._model = models.data[0].id
            else:
                self._model = self.options.model

            if self._model is None:
                raise ValueError("No model found. Please check your configurations. ")

    # Single-image input inference
    def run_single_image(self, image: Image.Image) -> str:
        
        ## Use base64 encoded image in the payload
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 =  base64.b64encode(buffered.getvalue()).decode("utf-8")

        chat_completion_from_base64 = self._client.chat.completions.create(
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.options.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }],
            model=self._model,
        )

        return chat_completion_from_base64.choices[0].message.content

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        result = self.run_single_image(high_res_image)

                        # del high_res_image
                        all_ocr_cells.append(
                            OcrCell(
                                id=0, # Only have one cell per OCR rect
                                text=result,
                                confidence=1.0, # TODO: add confidence score
                                bbox=BoundingBox.from_tuple(
                                    coord=(ocr_rect.l, ocr_rect.t, ocr_rect.r, ocr_rect.b),
                                    origin=CoordOrigin.TOPLEFT,
                                ),
                            )
                        )

                    # Post-process the cells
                    page.cells = self.post_process_cells(all_ocr_cells, page.cells)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page
