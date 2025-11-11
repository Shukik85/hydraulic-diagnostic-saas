"""
Authentication and authorization service
"""
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
import secrets
import structlog

from models.user import User, SubscriptionTier, SubscriptionStatus

logger = structlog.get_logger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return f"hyd_{secrets.token_urlsafe(32)}"

    async def create_user(self, email: str, password: str) -> User:
        """Create new user with free tier"""
        user = User(
            email=email,
            hashed_password=self.hash_password(password),
            subscription_tier=SubscriptionTier.FREE,
            subscription_status=SubscriptionStatus.TRIAL,
            api_key=self.generate_api_key()
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """Authenticate user by email and password"""
        from sqlalchemy import select

        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or not self.verify_password(password, user.hashed_password):
            return None

        return user
