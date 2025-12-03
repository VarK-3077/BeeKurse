import os
import httpx
import sqlite3
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
from pathlib import Path

# -------------------- sql injector -----------------------
import sys
sys.path.append("../test_debug")

from scripts.database_operations.sql_injector import add_subcategory_embedding_and_save
from config.config import Config

config = Config
# ---------------------------------------------------------

# -------------------- Demo Mode Config -------------------
DEMO_CONFIG_PATH = Path(__file__).parent.parent / "config" / "demo_config.json"
VENDOR_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "vendor_data" / "vendor_registry.json"

def load_demo_config():
    """Load demo configuration from JSON file"""
    try:
        with open(DEMO_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Demo config not found or invalid: {e}")
        return {"demo_mode": False}

DEMO_CONFIG = load_demo_config()
print(f"🎭 Demo mode: {DEMO_CONFIG.get('demo_mode', False)}")
# ---------------------------------------------------------


# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")
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

def get_demo_db_path(db_type: str) -> str:
    """Get the appropriate database path based on demo mode"""
    if DEMO_CONFIG.get("demo_mode", False):
        base_path = Path(__file__).parent.parent
        if db_type == "vendor":
            return str(base_path / DEMO_CONFIG["mock_databases"]["vendor_db"])
        elif db_type == "inventory":
            return str(base_path / DEMO_CONFIG["mock_databases"]["inventory_db"])
    # Non-demo mode paths
    if db_type == "vendor":
        return str(Path(__file__).parent.parent / "data" / "databases" / "sql" / "vendor.db")
    elif db_type == "inventory":
        return config.VENDOR_TEST_DB_PATH if config.USE_VENDOR_TEST_DB else config.SQL_DB_PATH
    return None

def add_to_vendor_registry(phone: str):
    """Add phone number to vendor_registry.json for WhatsApp validation"""
    try:
        # Normalize phone (remove spaces, dashes)
        normalized = ''.join(c for c in phone if c.isdigit() or c == '+')
        if normalized.startswith('+'):
            normalized = normalized[1:]  # Remove leading +

        # Load existing registry
        if VENDOR_REGISTRY_PATH.exists():
            with open(VENDOR_REGISTRY_PATH) as f:
                registry = json.load(f)
        else:
            registry = {"registered": []}

        # Add if not exists
        if normalized not in registry["registered"]:
            registry["registered"].append(normalized)
            with open(VENDOR_REGISTRY_PATH, 'w') as f:
                json.dump(registry, f, indent=2)
            print(f"📱 Added {normalized} to vendor registry")
        return True
    except Exception as e:
        print(f"Error updating vendor registry: {e}")
        return False

def create_vendor_in_sqlite(vendor_data: dict) -> dict:
    """Create vendor in SQLite database (for demo mode)"""
    db_path = get_demo_db_path("vendor")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    vendor_id = str(uuid.uuid4())
    cursor.execute('''INSERT INTO vendors
        (id, username, business_name, email, phone, business_type, address, description, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)''',
        (vendor_id, vendor_data["username"], vendor_data["business_name"],
         vendor_data["email"], vendor_data["phone"], vendor_data["business_type"],
         vendor_data.get("address"), vendor_data.get("description"),
         vendor_data["created_at"]))
    conn.commit()
    conn.close()

    return {**vendor_data, "id": vendor_id}

def get_vendor_from_sqlite(username: str) -> Optional[dict]:
    """Get vendor from SQLite database (for demo mode)"""
    db_path = get_demo_db_path("vendor")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vendors WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error fetching vendor from SQLite: {e}")
        return None

def get_vendor_by_email_sqlite(email: str) -> Optional[dict]:
    """Get vendor by email from SQLite database (for demo mode)"""
    db_path = get_demo_db_path("vendor")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vendors WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error fetching vendor by email from SQLite: {e}")
        return None

async def get_user_from_db(username: str):
    """Fetch user from JSON Server or SQLite (demo mode)"""
    # Demo mode: use SQLite
    if DEMO_CONFIG.get("demo_mode", False):
        vendor = get_vendor_from_sqlite(username)
        if vendor:
            # Need to add hashed_password for auth - fetch from JSON Server as fallback
            # For demo, we still use JSON Server for auth
            pass

    # Use JSON Server (primary)
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
    is_demo = DEMO_CONFIG.get("demo_mode", False)
    print(f"DEBUG: Registering vendor - {vendor.username} (demo_mode={is_demo})")

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

    # Create vendor in JSON Server (needed for auth in both modes)
    created_vendor = await create_user_in_db(vendor_dict)

    # Demo mode: also write to mock vendor.db
    if is_demo:
        try:
            create_vendor_in_sqlite(vendor_dict)
            print(f"🎭 Demo: Vendor saved to mock_vendor.db")
        except Exception as e:
            print(f"Warning: Could not save to mock vendor.db: {e}")

    # Add to vendor registry for WhatsApp validation
    add_to_vendor_registry(vendor.phone)

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
    # In demo mode, use mock_images folder under data/
    if DEMO_CONFIG.get("demo_mode", False):
        base_upload_dir = Path(__file__).parent.parent / DEMO_CONFIG.get("mock_upload_dir", "data/mock_images")
    else:
        base_upload_dir = Path(UPLOAD_DIR)

    vendor_dir = base_upload_dir / current_user.username
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
    # Build description with optional fields
    desc_parts = [product.description or ""]
    if product.care_instructions:
        desc_parts.append(f"care instruction: {product.care_instructions}")
    if product.materials:
        desc_parts.append(f"materials: {product.materials}")
    full_description = " | ".join(p for p in desc_parts if p)

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
        "descrption": full_description,
        "category": product.category,
        "imageid": image_files[0] if image_files else "",
        "subcategory": product.subcategory or "none",
        "rating": 0,
        "stock": product.stock,
    }

    # Check demo mode
    is_demo = DEMO_CONFIG.get("demo_mode", False)

    if is_demo:
        # Demo mode: write to mock inventory DB, skip VDB/KG
        db_path = get_demo_db_path("inventory")
        print(f"🎭 Demo mode: Saving product to mock database: {db_path}")

        # Generate short_id manually (skip embedding)
        import random
        import string
        short_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        complete_product["short_id"] = short_id

        # Direct SQL insert (no embeddings in demo mode)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO product_table
                (product_id, prod_name, store, category, subcategory, brand, colour,
                 description, dimensions, imageid, price, quantity, quantityunit,
                 rating, size, stock, short_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (complete_product["product_id"], complete_product["prod_name"],
                 complete_product["store"], complete_product["category"],
                 complete_product["subcategory"], complete_product["brand"],
                 complete_product["colour"], complete_product.get("descrption", ""),
                 complete_product["dimensions"], complete_product["imageid"],
                 float(complete_product["price"]) if complete_product["price"] else 0,
                 1, complete_product["qunatityunit"], 0,
                 complete_product["size"], complete_product["stock"], short_id))
            conn.commit()
            conn.close()
            print(f"🎭 Demo: Product saved with short_id={short_id}")
        except Exception as e:
            print(f"Error saving to mock DB: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save product: {str(e)}"
            )

        # Skip JSON Server in demo mode
        print("🎭 Demo: Skipping JSON Server, KG, and VDB writes")

    else:
        # Normal mode: use configured database with embeddings
        if config.USE_VENDOR_TEST_DB:
            db_path = config.VENDOR_TEST_DB_PATH
            print(f"🧪 Saving product to VENDOR TEST database: {db_path}")
        else:
            db_path = config.SQL_DB_PATH
            print(f"📦 Saving product to MAIN database: {db_path}")

        # Save to SQL database with embeddings
        saved_row = add_subcategory_embedding_and_save(complete_product, db_path=db_path)
        complete_product["short_id"] = saved_row.get("short_id")

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

    # Note: KG and VDB processing skipped in demo mode

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


@app.get("/products/me")
async def get_my_products(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get all products for the current vendor from the database"""

    # Determine which database to use based on demo mode
    is_demo = DEMO_CONFIG.get("demo_mode", False)
    if is_demo:
        db_path = get_demo_db_path("inventory")
        print(f"🎭 Demo: Fetching products from mock database")
    elif config.USE_VENDOR_TEST_DB:
        db_path = config.VENDOR_TEST_DB_PATH
    else:
        db_path = config.SQL_DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get products by store (business_name)
        cursor.execute(
            "SELECT * FROM product_table WHERE store = ?",
            (current_user.business_name,)
        )
        rows = cursor.fetchall()
        conn.close()

        products = []
        for row in rows:
            product = dict(row)
            # Remove embedding data from response (it's large)
            if 'subcategoryid' in product:
                del product['subcategoryid']
            products.append(product)

        return {"products": products, "total": len(products)}
    except Exception as e:
        print(f"Error fetching products: {e}")
        return {"products": [], "total": 0, "error": str(e)}


# Development server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🏪 Starting Vendor API Server")
    print("="*80)
    print(f"📍 API endpoint: http://0.0.0.0:8000")
    print(f"📦 Test DB enabled: {config.USE_VENDOR_TEST_DB}")
    if config.USE_VENDOR_TEST_DB:
        print(f"🧪 Using database: {config.VENDOR_TEST_DB_PATH}")
    else:
        print(f"📊 Using database: {config.SQL_DB_PATH}")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
