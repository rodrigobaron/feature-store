import copy
import datetime
import re
import types
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates

from . import backend as processor
from . import timeseries as ts
from . import utils

if TYPE_CHECKING:
    hybrid_property = property
else:
    from sqlalchemy.ext.hybrid import hybrid_property


Base = declarative_base()  # type: Any
valid_name = re.compile(r"^[a-zA-Z0-9\.#_-]+$")
partitions = Enum("year", "date", name="partition")


class FeatureStoreMixin(object):
    name = Column(String(128), primary_key=True, nullable=False)
    description = Column(String, default="")
    meta = Column(JSON, default={})
    version = Column(Integer, default=1, nullable=False)

    @validates("name")
    def _validate_name(self, key, value):
        if not valid_name.match(value):
            raise ValueError(f"Invalid name {value}")
        return value

    def as_dict(self):
        return {k: v if utils.is_jsonable(v) else str(v) for k, v in self.__dict__.items() if k[0] != "_"}

    def bump_version(self):
        if self.version:
            self.version += 1
        else:
            self.version = 1

    def update_from_dict(self, payload):
        if not payload:
            return
        if "name" in payload and self.name:
            raise ValueError(f"Cannot change name of {self.__class__.__name__}: use clone instead")
        for key, value in payload.items():
            if key == "meta" or key == "metadata":
                self.meta = copy.deepcopy(self.meta)
                
                # Merge old metadata with new
                if self.meta:
                    self.meta.update(value)
                else:
                    self.meta = value
                # Remove any keys that no longer have values
                self.meta = {k: v for k, v in self.meta.items() if v is not None}
            else:
                # Update fields
                setattr(self, key, value) if hasattr(self, key) else None
        self.bump_version()


class FeatureStoreVersion(Base):
    __tablename__ = "feature_store_version"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    version = Column(String(128), nullable=False)


class Namespace(Base, FeatureStoreMixin):
    __tablename__ = "namespace"

    storage = Column(String, nullable=False)
    partition = Column(partitions, default="date", nullable=False)

    @hybrid_property
    def namespace(self):
        return self.name

    @namespace.setter
    def namespace(self, value):
        self.name = value

    @namespace.expression
    def namespace(cls):
        return cls.name

    def clean(self):
        # Check for unused data and remove it
        active_feature_names = [f.name for f in self.features]
        feature_data = Storage(self.storage).ls()
        for feature in feature_data:
            if feature not in active_feature_names:
                # Redundant data... delete it
                store.delete(feature)


class Feature(Base, FeatureStoreMixin):
    __tablename__ = "feature"

    namespace = Column(String(128), ForeignKey("namespace.name"), primary_key=True)
    namespace_object = relationship("Namespace", backref="features")

    serialized = Column(Boolean, default=False, nullable=False)
    transform = Column(JSON, nullable=True)
    check = Column(String(), nullable=False)

    @hybrid_property
    def full_name(self):
        return f"{self.namespace}/{self.name}"

    @validates("serialized")
    def validate_serialized(self, key, value):
        if self.serialized is not None and value != self.serialized:
            raise ValueError("Cannot change serialized setting on existing feature")
        return value

    @validates("transform")
    def validate_transform(self, key, value):
        if not value:
            return value
        if isinstance(value.get("function"), str):
            # Function already serialized, no conversion required
            func = value.get("function")
        elif isinstance(value.get("function"), types.FunctionType):
            func = utils.serialize(value["function"])
        else:
            raise ValueError("Transform must be a Python function, accepting a single dataframe input")
        assert "function" in value.keys(), "Transform must have a function defined"
        assert "args" in value.keys(), "Transform must have arguments defined"
        # Convert function to base64/cloudpickle format
        return {
            "format": "cloudpickle",
            "function": func,
            "args": value["args"],
        }

    @classmethod
    def clone_from(cls, other, namespace, name):
        if not isinstance(other, cls):
            raise ValueError(f"Must clone from another {cls.__name__}")
        clone = cls()
        # Build new Feature with same settings as old
        clone.namespace = namespace
        clone.name = name
        payload = other.as_dict()
        payload.pop("namespace")
        payload.pop("name")
        payload.pop("version")
        clone.update_from_dict(payload)
        return clone

    def save(self, df, store):
        store.save(self.namespace, df, partition=self.namespace_object.partition, serialized=self.serialized)

    def load_transform(self, storage, from_date, to_date, freq, time_travel, last=False, callers=[]):
        # Get the SQLAlchemy session for this feature
        session = sa.inspect(self).session
        if not session:
            raise RuntimeError(f"{self.name} is not bound to an SQLAlchemy session")
        # Check for recursive transforms
        if self.full_name in callers:
            raise RuntimeError(f"Recursive feature transform detected on {self.full_name}")
        # Load the transform function
        func = utils.deserialize(self.transform["function"])
        # Load the features to transform
        dfs = []
        # Load each requested feature
        for f in self.transform["args"]:
            namespace, name = f.split("/")[0], "/".join(f.split("/")[1:])
            feature = session.query(Feature).filter_by(name=name, namespace=namespace).one_or_none()
            if not feature:
                raise ValueError(f"No feature named {name} exists in {namespace}")
            # Load individual feature
            df = feature.load(
                storage,
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                last=last,
                callers=[*callers, self.full_name],
            )
            dfs.append(df.rename(columns={name: f"{namespace}/{name}"}))
        # Merge features into a single dataframe
        dfs = ts.concat(dfs)
        # Make sure columns are in the same order as args
        dfs = dfs[self.transform["args"]]
        # Apply transform function
        transformed = ts.transform(dfs, func)
        return transformed

    def load(
        self,
        store,
        from_date=None,
        to_date=None,
        freq=None,
        time_travel=None,
        last=False,
        callers=[],
    ):
        # Does this feature need to be transformed?
        if self.transform:
            return self.load_transform(
                store,
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                last=last,
                callers=callers,
            )
        
        # Restrict which partitions are loaded when getting last value
        if last:
            from_date = store.last(self.name)
            to_date = None
        
        # Load dataframe
        return store.load(
            self.namespace,
            from_date=from_date,
            to_date=to_date,
            freq=freq,
            time_travel=time_travel,
            serialized=self.serialized,
        )

    # def last(self):
    #     # Fetch last feature value
    #     df = self.load(last=True)
    #     result = df.tail(1)
    #     if result.empty:
    #         return None
    #     else:
    #         return result["value"].iloc[0]

    def delete_data(self, storage):
        # Deletes all of the data on this feature
        store = self.namespace_object._backend(storage)
        store.delete(self.name)

    def import_data_from(self, other, storage):
        # Copy data over from another feature
        if not isinstance(other, self.__class__):
            raise ValueError(f"Must clone from another {other.__name__}")
        # Get location of other feature to copy from
        store_from = other.namespace_object._backend(storage)
        store_to = self.namespace_object._backend(storage)
        # Copy data to new location
        store_from.copy(other.name, self.name, store_to)
