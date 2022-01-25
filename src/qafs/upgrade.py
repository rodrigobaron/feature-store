from . import connection as conn
from . import model as model
from . import version as version
from alembic.operations import Operations
from alembic.migration import MigrationContext
from sqlalchemy import String, Column
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from packaging.version import parse


def upgrade(engine):
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    r = session.query(model.FeatureStoreVersion).order_by(model.FeatureStoreVersion.timestamp.desc()).first()
    if not r:
        current_version = "0"
    else:
        current_version = r.version

    if parse(current_version) >= parse(version.__version__):
        # Up-to-date: nothing to do
        session.close()
        return

    if parse(current_version) < parse(version.__version__):
        print(f"Upgrading database schema...")
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            op = Operations(context)
            try:
                op.add_column("namespace", Column("backend", String(128)))
            except OperationalError:
                pass

    # Add version number
    obj = model.FeatureStoreVersion()
    obj.version = version.__version__
    session.add(obj)
    session.commit()

    session.close()
