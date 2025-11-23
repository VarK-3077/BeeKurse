import os
import httpx
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Optional, List
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from jose import JWTError, jwt
from passlib.context import CryptContext

import uuid
from datetime import datetime
from fastapi import Form
import json

# -------------------- sql injector -----------------------
import sys
sys.path.append("../test_debug")

from sql_injector import add_subcategory_embedding_and_save
# ---------------------------------------------------------


# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
JSON_SERVER_URL = os.getenv("JSON_SERVER_URL", "http://localhost:3001")

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_files")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: Optional[str] = None
    username: str
    email: EmailStr
    business_name: str
    phone: str
    business_type: str
    address: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

class UserInDB(User):
    hashed_password: str

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: EmailStr
    business_name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., pattern=r'^\+?[\d\s\-\(\)]+$')
    business_type: str = Field(..., description="Type of business (clothing, textiles, accessories, etc.)")
    address: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)

class UserLogin(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str

class VendorResponse(BaseModel):
    id: str
    username: str
    email: str
    business_name: str
    phone: str
    business_type: str
    address: Optional[str] = None
    description: Optional[str] = None
    is_active: bool
    created_at: datetime

class VendorProfileUpdate(BaseModel):
    email: Optional[EmailStr] = None
    business_name: Optional[str] = Field(None, min_length=1, max_length=200)
    phone: Optional[str] = Field(None, pattern=r'^\+?[\d\s\-\(\)]+$')
    business_type: Optional[str] = None
    address: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)

# Pydantic model for product data
class ProductCreate(BaseModel):
    name: str
    category: str
    subcategory: str
    price: str
    description: Optional[str] = None
    colors: List[str] = []
    sizes: List[str] = []
    materials: Optional[str] = None
    care_instructions: Optional[str] = None
    stock: int
    dimensions: Optional[str] = None
    brand: Optional[str] = None
    unit: Optional[str] = "item"

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user_from_db(username: str):
    """Fetch user from JSON Server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{JSON_SERVER_URL}/users", params={"username": username})
            users = response.json()
            if users and len(users) > 0:
                return UserInDB(**users[0])
            return None
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None

async def get_user_by_email(email: str):
    """Fetch user by email from JSON Server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{JSON_SERVER_URL}/users", params={"email": email})
            users = response.json()
            if users and len(users) > 0:
                return UserInDB(**users[0])
            return None
        except Exception as e:
            print(f"Error fetching user by email: {e}")
            return None

async def create_user_in_db(vendor_data: dict):
    """Create vendor in JSON Server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{JSON_SERVER_URL}/users",
                json=vendor_data
            )
            return response.json()
        except Exception as e:
            print(f"Error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user"
            )

async def update_user_in_db(user_id: str, update_data: dict):
    """Update vendor in JSON Server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(
                f"{JSON_SERVER_URL}/users/{user_id}",
                json=update_data
            )
            return response.json()
        except Exception as e:
            print(f"Error updating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not update user"
            )

async def authenticate_user(username: str, password: str):
    user = await get_user_from_db(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = await get_user_from_db(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"content": "Hello world"}

@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=VendorResponse)
async def register(vendor: UserRegister):
    """Register a new vendor with complete business information"""
    print(f"DEBUG: Registering vendor - {vendor.username}")

    # Check if username already exists
    existing_user = await get_user_from_db(vendor.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Check if email already exists
    existing_email = await get_user_by_email(vendor.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Prepare vendor data
    vendor_dict = vendor.dict(exclude={"password"})
    vendor_dict["hashed_password"] = get_password_hash(vendor.password)
    vendor_dict["created_at"] = datetime.utcnow().isoformat()
    vendor_dict["is_active"] = True

    # Create vendor in database
    created_vendor = await create_user_in_db(vendor_dict)

    # Return vendor response
    return VendorResponse(
        id=str(created_vendor.get("id")),
        username=created_vendor["username"],
        email=created_vendor["email"],
        business_name=created_vendor["business_name"],
        phone=created_vendor["phone"],
        business_type=created_vendor["business_type"],
        address=created_vendor.get("address"),
        description=created_vendor.get("description"),
        is_active=created_vendor["is_active"],
        created_at=datetime.fromisoformat(created_vendor["created_at"])
    )

@app.get("/users/me", response_model=VendorResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get current vendor profile"""
    return VendorResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        business_name=current_user.business_name,
        phone=current_user.phone,
        business_type=current_user.business_type,
        address=current_user.address,
        description=current_user.description,
        is_active=current_user.is_active,
        created_at=current_user.created_at or datetime.utcnow()
    )

@app.patch("/users/me", response_model=VendorResponse)
async def update_vendor_profile(
    profile_update: VendorProfileUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Update vendor profile information"""

    # Prepare update data (only include fields that are not None)
    update_data = profile_update.dict(exclude_none=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )

    # If email is being updated, check if it's already taken
    if "email" in update_data:
        existing_email = await get_user_by_email(update_data["email"])
        if existing_email and existing_email.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )

    # Update user in database
    updated_vendor = await update_user_in_db(current_user.id, update_data)

    return VendorResponse(
        id=str(updated_vendor.get("id")),
        username=updated_vendor["username"],
        email=updated_vendor["email"],
        business_name=updated_vendor["business_name"],
        phone=updated_vendor["phone"],
        business_type=updated_vendor["business_type"],
        address=updated_vendor.get("address"),
        description=updated_vendor.get("description"),
        is_active=updated_vendor["is_active"],
        created_at=datetime.fromisoformat(updated_vendor["created_at"])
    )


# Updated endpoint
@app.post("/files/")
async def upload_files(
    files: list[UploadFile],
    current_user: Annotated[User, Depends(get_current_active_user)],
    product_data: str = Form(...)  # JSON string of product data
):
    """Upload product files (images, documents) for the vendor"""

    # Parse product data
    try:
        product_dict = json.loads(product_data)
        product = ProductCreate(**product_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid product data: {str(e)}"
        )

    # Create vendor-specific directory
    vendor_dir = os.path.join(UPLOAD_DIR, current_user.username)
    os.makedirs(vendor_dir, exist_ok=True)

    uploaded_files = []
    image_files = []

    for file in files:
        # Create unique filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(vendor_dir, filename)

        # Save file
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        file_info = {
            "filename": file.filename,
            "saved_as": filename,
            "size": len(content),
            "content_type": file.content_type
        }
        uploaded_files.append(file_info)

        # Track image files separately
        if file.content_type and file.content_type.startswith("image/"):
            image_files.append(filename)

    # Generate product_id using uuid
    product_id = f"PID-{str(uuid.uuid4())}"

    # Prepare complete product data
    complete_product = {
        "product_id": product_id,
        "prod_name": product.name,
        "price": product.price,
        "quantity": "1",
        "qunatityunit": product.unit,
        "size": ", ".join(product.sizes) if product.sizes else "",
        "store": current_user.business_name,
        "dimensions": product.dimensions or "",
        "brand": product.brand or "",
        "colour": product.colors[0] if product.colors else "",
        "descrption": product.description + " | care instruction: " + product.care_instructions + " | materials: " + product.materials or "",
        "category": product.category,
        "imageid": image_files[0] if image_files else "",
        "subcategory": product.subcategory or "none",
        "rating": 0,
        "stock": product.stock,
        # "materials": product.materials or "",
        # "care_instructions": product.care_instructions or "",
        # "vendor_username": current_user.username,
        # "created_at": datetime.utcnow().isoformat()
    }

    add_subcategory_embedding_and_save(complete_product)

    # Save product to JSON server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{JSON_SERVER_URL}/products",
                json=complete_product
            )
            response.raise_for_status()
            saved_product = response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save product to database: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving product: {str(e)}"
        )

    # TODO: Save complete_product to database
    # TODO: Process files for vector database, knowledge graph, etc.

    return {
        "message": "Product uploaded successfully",
        "vendor": current_user.username,
        "business_name": current_user.business_name,
        "product": complete_product,
        "files": uploaded_files,
        "total_files": len(uploaded_files)
    }


@app.get("/vendors/", response_model=list[VendorResponse])
async def list_vendors(
    current_user: Annotated[User, Depends(get_current_active_user)],
    business_type: Optional[str] = None,
    limit: int = 50
):
    """List all vendors (with optional filtering by business type)"""
    async with httpx.AsyncClient() as client:
        try:
            params = {"_limit": limit}
            if business_type:
                params["business_type"] = business_type

            response = await client.get(f"{JSON_SERVER_URL}/users", params=params)
            vendors = response.json()

            return [
                VendorResponse(
                    id=str(v.get("id")),
                    username=v["username"],
                    email=v["email"],
                    business_name=v["business_name"],
                    phone=v["phone"],
                    business_type=v["business_type"],
                    address=v.get("address"),
                    description=v.get("description"),
                    is_active=v.get("is_active", True),
                    created_at=datetime.fromisoformat(v["created_at"])
                )
                for v in vendors
            ]
        except Exception as e:
            print(f"Error listing vendors: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not fetch vendors"
            )

@app.get("/vendors/{username}", response_model=VendorResponse)
async def get_vendor_by_username(
    username: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get a specific vendor's public profile"""
    vendor = await get_user_from_db(username)
    if not vendor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vendor not found"
        )

    return VendorResponse(
        id=str(vendor.id),
        username=vendor.username,
        email=vendor.email,
        business_name=vendor.business_name,
        phone=vendor.phone,
        business_type=vendor.business_type,
        address=vendor.address,
        description=vendor.description,
        is_active=vendor.is_active,
        created_at=vendor.created_at or datetime.utcnow()
    )
