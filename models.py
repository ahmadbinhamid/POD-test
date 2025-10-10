import os
from pydantic import BaseModel, Field, HttpUrl, FilePath, FileUrl, field_validator
import mimetypes
from pathlib import Path
from typing import Optional, Union, Dict, List
from datetime import date
from PIL import Image


class InterVL2Request(BaseModel):
    image_url_or_path_or_base64:str
    prompt:str

class InterVL2Response(BaseModel):
    response:str

class BillOfLadings(BaseModel):
    bill_no: str = Field("", description="Unique bill of lading number")
    date: Optional[str] = Field(None, description="The date from the bill of lading")

    @field_validator("bill_no", mode="before")
    @classmethod
    def _cast_bill_no_to_str(cls, v):
        # if it's None, default to empty string; otherwise str() it
        if v is None:
            return ""
        return str(v)

class CarrierInfo(BaseModel):
    carrier_name: str = Field("", description="Name of the carrier")

class Stamp(BaseModel):
    pod_date: Optional[str] = Field(
        None, description="Date of the POD, in YYYY-MM-DD format"
    )
    pod_sign: Optional[str] = Field(
        None, description="Whether signature was present or not ('yes', 'no')"
    )
    stamp_exist: Optional[str] = Field(
        None, description="Whether stamp was present or not ('yes', 'no')"
    )
    damage: Optional[Union[int, str]] = Field(
        None, description="Number of items damaged (non-negative integer or '-')"
    )
    over: Optional[Union[int, str]] = Field(
        None, description="Number of items over (non-negative integer or '-')"
    )
    short: Optional[Union[int, str]] = Field(
        None, description="Number of items short (non-negative integer or '-')"
    )
    refused: Optional[Union[int, str]] = Field(
        None, description="Number of items refused (non-negative integer or '-')"
    )
    seal_intact: Optional[str] = Field(
        None, description="Whether the seal was intact ('yes', 'no')"
    )
    damaged_kept: Optional[Union[int, str]] = Field(
        None, description="Number of damaged cartons kept (non-negative integer or '-')"
    )
    roc_damaged: Optional[Union[int, str]] = Field(
        None, description="Record or acknowledgment of damage (non-negative integer or '-')"
    )
    total_received: Optional[Union[int, str]] = Field(
        None, description="Number of cartons received (non-negative integer or '-')"
    )

class CustomerOrderInfo(BaseModel):
    total_order_quantity: int = Field(0, description="Total number of packages handled")


class Signatures(BaseModel):
    
    receiver_signature: str = Field(
        default=None,
        pattern="^(yes|no)$",
        description="Indicates whether the receiver has provided a signature. Accepts 'yes' or 'no'."
    )

class DeliveryReceipt(BaseModel):

    
    pod_date: Optional[str] = Field(
        None, description="Date of the POD, in YYYY-MM-DD format"
    )
    pod_sign: Optional[str] = Field(
        None, description="Whether signature was present or not ('yes', 'no')"
    )
    total_received: Optional[Union[List[int] ,int, str]] = Field(
        None, description="Number of cartons received (non-negative integer or 'empty')"
    )
    refused: Optional[Union[List[int], int, str]] = Field(
        None, description="Number of items refused (non-negative integer or 'empty')"
    )
    damage: Optional[Union[List[int], int, str]] = Field(
        None, description="Number of items damaged (non-negative integer or 'empty')"
    )
    customer_order_num: Optional[Union[List[int], List[str], int, str]] = Field(
        None, description="Customer number written on the receipt (non-negative integer or 'empty')"
    )


class OCRRequest(BaseModel):
    id: str = Field(..., alias="_id")
    file_url_or_path: str = Field(
        ..., description="URL or file path to the OCR"
    )

    @field_validator("file_url_or_path")
    def validate_file_url_or_path(cls, value):
        # If it's a URL, check its MIME type
        if isinstance(value, HttpUrl):
            # Allow only common image extensions in the URL
            if not value.path.lower().endswith((".pdf", ".PDF")):
                raise ValueError("URL must point to a valid pdf file.")
        elif isinstance(value, Path):
            # Validate file existence and MIME type for a local file
            if not os.path.exists(value):
                raise ValueError(f"The file path '{value}' does not exist.")
            mime_type, _ = mimetypes.guess_type(value)
            if mime_type is None or not mime_type.startswith("application/pdf"):
                raise ValueError(
                    f"The file path '{value}' must point to a valid pdf file."
                )
        return value
        
    # class Config:
    #     allow_population_by_field_name = True  

class OCRResponse(BaseModel):
    bill_of_lading: BillOfLadings = Field(
        BillOfLadings(), description="Details of the bill of lading"
    )
    # carrier: CarrierInfo = Field(CarrierInfo(), description="Details of the carrier")
    stamp: Stamp = Field(Stamp(), description="Details of the stamp")
    customer_order_info: CustomerOrderInfo = Field(
        CustomerOrderInfo(), description="Details of the customer order"
    )
    signatures: Signatures = Field(
        Signatures(), description="Details of the signatures"
    )
    


class Regions(BaseModel):
    bill_of_lading: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing Bill of Ladings"
    )
    # carrier: Optional[Image.Image] = Field(
    #     None, description="Region containing carrier info"
    # )
    stamp: Optional[Union[Image.Image, str]] = Field(None, description="Region containing stamp")
    customer_order_info: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing order info"
    )
    signatures: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )

    class Config:
        arbitrary_types_allowed = True

# {
    # "pod_date": "10/23/2024",
    # "pod_sign": "yes",
    # "total_received": 205,
    # "damage": "empty",
    # "refused": "empty"
    # }

class OCRBatch(BaseModel):
    image: str
    prompt: str

class OCRBatch(BaseModel):
    page_type: str
    region_name: str
    prompt: str
    image: str
    stamp_exist:Optional[str]=None

class OCRBatchResponse(BaseModel):
    page_type: str
    region_name: str
    ocr_response: dict
    stamp_exist:Optional[str]=None

class ReceiptRegions(BaseModel):
    pod_date: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing date"
    )
    # carrier: Optional[Image.Image] = Field(
    #     None, description="Region containing carrier info"
    # )
    customer_order_num: Optional[Union[Image.Image, str]] = Field(None, description="Region containing stamp")
    total_received: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing order info"
    )
    damage: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )
    refused: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )
    pod_sign: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )

    class Config:
        arbitrary_types_allowed = True


class PODResponse(BaseModel):
    B_L_Number: Optional[Union[str, int]] = None
    Stamp_Exists: Optional[str] = None
    Seal_Intact: Optional[str] = None
    POD_Date: Optional[str] = None
    Signature_Exists: Optional[str] = None
    Issued_Qty: Optional[Union[str, int]] = None
    Received_Qty: Optional[Union[str, int]] = None
    Damage_Qty: Optional[Union[str, int]] = None
    Short_Qty: Optional[Union[str, int]] = None
    Over_Qty: Optional[Union[str, int]] = None
    Refused_Qty: Optional[Union[str, int]] = None
    Customer_Order_Num: Optional[Union[str, int, List[Union[str, int]]]] = None  # Accept list too
    id: str = Field(..., alias="_id")


    class Config:
        populate_by_name = True