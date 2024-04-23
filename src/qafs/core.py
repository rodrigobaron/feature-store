import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd
import pyspark
from pyspark.sql import SparkSession

import pandera
from pandera import Column, DataFrameSchema, io

from . import connection as conn
from .storage import LocalStorage
from . import model
from . import timeseries as ts
from . import upgrade as upgrade
from . import utils
from .exceptions import FeatureStoreException, MissingFeatureException
from .backend import SparkBackend

DEFAULT_NAMESPACE = 'base'

class FeatureStore(object):
    """**QAFS Feature Store**"""

    def __init__(self, project, spark_session=None, verbose=False):
        """
        Args:
            storage (QAFSStorage): Storage manager.
            db_connection (str): SQLAlchemy connection string for database.
            backend (str): DataFrame backend.
            verbose (bool): Logging level. 
        """
        project_path = f'./.qafs/{project}'
        db_connection = "/".join(["sqlite://", project_path, 'fs.db'])
        self.storage = LocalStorage(path=f'{project_path}/storage')
        os.makedirs(project_path, exist_ok=True)
        self.spark_session = spark_session or SparkSession.builder.getOrCreate()
        self.backend = SparkBackend(self.storage, self.spark_session)
        self.engine, self.session_maker = conn.connect(db_connection)
        model.Base.metadata.create_all(self.engine)
        upgrade.upgrade(self.engine)
        utils.setup_logging(level=logging.INFO if verbose else logging.WARNING)

    def _list(self, cls, name=None, namespace=None, regex=None, friendly=True):
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            
            # Filter by namespace
            if namespace:
                r = r.filter_by(namespace=namespace)
            
            # Filter by matching name
            if name:
                r = r.filter_by(name=name)
            objects = r.all()
            
            df = pd.DataFrame([obj.as_dict() for obj in objects])
            if df.empty:
                return pd.DataFrame()
            
            # Filter by regex search on name
            if regex:
                df = df[df.name.str.contains(regex)]
            
            # List transforms as simple true/false
            if "transform" in df.columns and friendly:
                df = df.assign(transform=df["transform"].apply(lambda x: x is not None))
            
            # Sort the columns
            column_order = ["namespace", "name", "version", "description", "meta"]
            
            column_order = [c for c in column_order if c in df.columns]
            df = df[[*column_order, *df.columns.difference(column_order)]]
            return df

    def list_namespaces(
        self, name: Optional[str] = None, namespace: Optional[str] = None, regex: Optional[str] = None
    ):
        """List namespaces in the feature store.

        Search by name or regex query.

        Parameters
        ----------
        name: str, optional
            name of namespace to filter by.
        namespace: str, optional
            same as name.
        regex: str, optional
            regex filter on name.

        Returns
        -------
        pd.DataFrame
            DataFrame of namespaces and metadata.
        """
        return self._list(model.Namespace, name=name if name is not None else namespace, regex=regex)


    def create_namespace(
        self, name: str, storage: Optional[LocalStorage] = None, partition: Optional[str] = None, description: Optional[str] = None, meta: Optional[Dict] = {}
    ):
        """Create a new namespace in the feature store.

        Parameters
        ----------
        name: str
            name of the namespace.
        description: str, optional
            description for this namespace.
        partition: str, optional
            partitioning of stored timeseries (default: `"date"`).
        meta: dict, optional
            key/value pairs of metadata.
        """
        with conn.session_scope(self.session_maker) as session:
            storage = storage or self.storage
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            if not r.one_or_none():
                obj = model.Namespace()
                obj.update_from_dict({"name": name, "storage": storage.path, "partition": partition, "description": description, "meta": meta})
                session.add(obj)

    def update_namespace(self, name: str, description: Optional[str] = None, meta: Optional[Dict] = {}):
        """Update a namespace in the feature store.

        Parameters
        ----------
        name: str
            namespace to update.
        description: str, optional
            updated description.
        meta: dict, optional
            updated key/value pairs of metadata.
            To remove metadata, update using `{"key_to_remove": None}`.
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(f"No existing {model.Namespace.__name__} named {name}")
            obj.update_from_dict(
                {
                    "description": description,
                    "meta": meta,
                }
            )

    def delete_namespace(self, name: str, delete_data: bool = False):
        """Delete a namespace from the feature store.

        Parameters
        ----------
        name: str
            namespace to be deleted.
        delete_data: bool
            also delete feature data
        """
        if not self.list_features(namespace=name).empty:
            raise FeatureStoreException(f"{name} still contains features: these must be deleted first")
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(f"No existing {model.Namespace.__name__} named {name} in {name}")
            if hasattr(obj, "delete_data") and delete_data:
                obj.delete_data()
            session.delete(obj)

    def clean_namespace(self, name: str):
        """Removes any data that is not associated with features in the namespace.

        Run this to free up disk space after deleting features

        Parameters
        ----------
        name: str
            namespace to clean
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            namespace = r.one_or_none()
            if not namespace:
                raise MissingFeatureException(f"No existing Namespace named {name}")
            namespace.clean(self.storage)

    def list_features(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        regex: Optional[str] = None,
        friendly: Optional[bool] = None,
    ):
        """List features in the feature store.

        Search by namespace, name and/or regex query

        Parameters
        ----------
        name: str, optional
            name of feature to filter by.
        namespace: str, optional
            namespace to filter by.
        regex: str, optional
            regex filter on name.
        friendly: bool, optional
            simplify output for user.

        Returns
        -------
        pd.DataFrame
            DataFrame of features and metadata.
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Feature)

            # Filter by namespace
            if namespace:
                r = r.filter_by(namespace=namespace)
            
            # Filter by matching name
            if name:
                r = r.filter_by(name=name)
            
            objects = r.all()
            df = pd.DataFrame([obj.as_dict() for obj in objects])
            
            if df.empty:
                return pd.DataFrame()
            
            # Filter by regex search on name
            if regex:
                df = df[df.name.str.contains(regex)]
            
            # List transforms as simple true/false
            if "transform" in df.columns and friendly:
                df = df.assign(transform=df["transform"].apply(lambda x: x is not None))
            
            # Sort the columns
            column_order = ["namespace", "name", "version", "description", "meta"]
            column_order = [c for c in column_order if c in df.columns]
            
            df = df[[*column_order, *df.columns.difference(column_order)]]
            return df

    def create_feature(
        self,
        name: str,
        check: Column,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        serialized: Optional[bool] = None,
        transform: Optional[str] = None,
        meta: Optional[Dict] = None,
    ):
        """Create a new feature in the feature store.

        Parameters
        ----------
        name: str
            name of the feature
        check: Column
            Data validation check
        namespace: str, optional
            namespace which should hold this feature.
        description: str, optional
            description for this namespace.
        serialized: bool, optional
            if `True`, converts values to JSON strings before saving,
            which can help in situations where the format/schema of the data changes
            over time.
        transform: str, optional
            pickled function code for feature transforms.
        meta: dict, optional
            key/value pairs of metadata.
        """
        meta = meta if meta is not None else {}
        namespace = namespace or DEFAULT_NAMESPACE

        ls = self._list(model.Namespace, namespace=namespace)
        if ls.empty:
            raise MissingFeatureException(f"{namespace} namespace does not exist")

        check._name = name
        check_yaml = io.to_yaml(DataFrameSchema({name: check}))

        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Feature)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            # Update transformation
            if r.one_or_none():
                return self.update_feature(
                    name=name,
                    namespace=namespace,
                    description=description,
                    transform=transform,
                    check=check,
                    meta=meta,
                )

            obj = model.Feature()
            obj.update_from_dict(
                {
                    "name": name,
                    "namespace": namespace,
                    "description": description,
                    "serialized": serialized,
                    "transform": transform,
                    "check": check_yaml,
                    "meta": meta,
                }
            )
            session.add(obj)

    def clone_feature(
        self,
        name: str,
        namespace: Optional[str] = None,
        from_name: Optional[str] = None,
        from_namespace: Optional[str] = None,
    ):
        """Create a new feature by cloning an existing one.

        Parameters
        ----------
        name: str
            name of the feature.
        namespace: str, optional
            namespace which should hold this feature.
        from_name: str
            the name of the existing feature to copy from.
        from_namespace: str, optional
            namespace of the existing feature.
        """
        if self.list_namespaces(namespace=from_namespace).empty:
            raise MissingFeatureException(f"{from_namespace} namespace does not exist")

        if self.list_namespaces(namespace=to_namespace).empty:
            raise MissingFeatureException(f"{to_namespace} namespace does not exist")

        with conn.session_scope(self.session_maker) as session:
            # Get the existing feature
            r = session.query(model.Feature)
            r = r.filter_by(namespace=from_namespace, name=from_name)
            feature = r.one_or_none()
 
            if not feature:
                raise MissingFeatureException(f"No existing Feature named {from_name} in {from_namespace}")
 
            # Create the new feature
            new_feature = model.Feature.clone_from(feature, to_namespace, to_name)
            session.add(new_feature)
 
            # Copy data to new feature, if this raises exception will rollback
            if not new_feature.transform:
                new_feature.import_data_from(feature, self.storage)

    def delete_feature(self, name: str, namespace: Optional[str] = None, delete_data: Optional[bool] = False):
        """Delete a feature from the feature store.

        Parameters
        ----------
        name: str
            name of feature to delete.
        namespace str, optional
            namespace, if not included in feature name.
        delete_data: bool, optional
            if set to `True` will delete underlying stored data
            for this feature, otherwise default behaviour is to delete the feature store
            metadata but leave the stored timeseries values intact.
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Feature)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(f"No existing {model.Feature.__name__} named {name} in {namespace}")
            if hasattr(obj, "delete_data") and delete_data:
                obj.delete_data(self.storage)
            session.delete(obj)

    def save(self, df: Union[pd.DataFrame, pyspark.sql.DataFrame], features: Optional[List[str]] = None, namespace: Optional[str] = None, index: Optional[str] = None):
        """Save a DataFrame of feature values to the feature store.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of feature values.
            Must have a `time` column or DateTimeIndex of time values.
            Optionally include a `created_time` column (defaults to `utcnow()` if omitted).
            For a single feature: a `value` column, or column header of feature `namespace/name`.
            For multiple features name the columns using `namespace/name`.
        name: str, optional
            name of feature, if not included in DataFrame column name.
        namespace: str, optional
            namespace, if not included in DataFrame column name.
        """
        feature_columns = df.columns.difference(["time", "created_time"])

        if len(feature_columns) == 1:
            if features is None:
                feature = feature_columns[0]
            
            namespace = namespace or DEFAULT_NAMESPACE
            name = features[0]

            if self.list_features(name=name).empty:
                raise MissingFeatureException(f"Feature named {name} does not exist in {namespace}")

            # Save data for this feature
            with conn.session_scope(self.session_maker) as session:
                feature = session.query(model.Feature).filter_by(name=name, namespace=namespace).one()

                # Save individual feature
                check = io.from_yaml(feature.check)
                feature_check = check.columns[name]

                try:
                    df_columns = list(df.columns)
                    if feature_check.name not in df_columns:
                        name = f"{namespace}/{name}"
                        feature_check._name = name
                    df = feature_check.validate(df)
                except pandera.errors.SchemaError as pse:
                    if self.check_raise_error:
                        raise pse
                    else:
                        logging.error(str(pse))

                # df = df.rename(columns={name: f"{namespace}/{name}"})
                feature.save(df, self.backend)
        else:
            if feature is not None:
                feature_df = df[[*df.columns.difference(feature_columns), feature]]
                self.save(feature_df, feature=feature)
            else:
                for feature_name in feature_columns:
                    # Save individual features
                    feature_df = df[[*df.columns.difference(feature_columns), feature_name]]
                    self.save(feature_df, name=feature_name)

    def load_features(
        self,
        features: Union[str, list, pd.DataFrame],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        freq: Optional[str] = None,
        time_travel: Optional[str] = None,
    ):
        """Load a DataFrame of feature values from the feature store.

        Parameters
        ----------
        features: Union[str, list, pd.DataFrame]
            name of feature to load, or list/DataFrame of feature namespaces/name.
        from_date: datetime, optional
            start date to load timeseries from, defaults to everything.
        to_date datetime, optional
            end date to load timeseries to, defaults to everything.
        freq: str, optional
            frequency interval at which feature values should be sampled.
        time_travel str, optional
            timedelta string, indicating that time-travel should be applied to the
            returned timeseries values, useful in forecasting applications.

        Returns
        -------
        Union[pd.DataFrame, dask.DataFrame]
            depending on which backend was specified in the feature store.
        """
        dfs = []
        # Load each requested feature
        for f in features:
            name_splited = f.split('/')
            namespace = name_splited[0] if len(name_splited) > 1 else DEFAULT_NAMESPACE
            name = name_splited[-1]
            with conn.session_scope(self.session_maker) as session:
                feature = session.query(model.Feature).filter_by(name=name, namespace=namespace).one_or_none()
                if not feature:
                    raise MissingFeatureException(f"No feature named '{name}' exists in '{namespace}'")

                # Load individual feature
                df = feature.load(
                    self.backend,
                    from_date=from_date,
                    to_date=to_date,
                    freq=freq,
                    time_travel=time_travel,
                )
                dfs.append(df.rename(columns={name: f"{namespace}/{name}"}))
        return ts.concat(dfs)

    def update_feature(
        self,
        name: str,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        transform: Optional[str] = None,
        check: Optional[Column] = None,
        meta: Optional[Dict] = {},
    ):
        """Update a feature in the feature store.

        Parameters
        ----------
        name: str
            feature to update.
        namespace: str, optional
            namespace, if not included in feature name.
        description: str, optional
            updated description.
        transform: str, optional
            pickled function code for feature transforms.
        check: Column, optional
            data validation check.
        meta: dict, optional
            updated key/value pairs of metadata.
            To remove metadata, update using `{"key_to_remove": None}`.
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Feature)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(f"No existing {model.Feature.__name__} named {name} in {namespace}")

            to_update = {
                "description": description,
                "transform": transform,
                "meta": meta,
            }

            check_yaml = None
            if check is not None:
                check._name = name
                check_yaml = io.to_yaml(DataFrameSchema({name: check}))
                to_update = {"check": check_yaml, **to_update}

            obj.update_from_dict(to_update)

    def transform(self, name: str, check: Column, t_namespace: Optional[str] = None, from_features: List = []):
        """Decorator for creating/updating virtual (transformed) features.

        Use this on a function that accepts a dataframe input and returns an output dataframe
        of tranformed values.

        Parameters
        ----------
        name: str
            feature to update.
        check: Column
            data validation check.
        t_namespace: str, optional
            namespace, if not included in feature name.
        from_features: list
            list of features which should be transformed by this one
        """

        def decorator(func):
            # Create or update feature with transform
            computed_from = [str(f) for f in from_features]
            for feature in from_features:
                feature_split = feature.split('/')
                feature_namespace = feature_split[0] if len(feature_split) > 1 else DEFAULT_NAMESPACE
                feature_name = feature_split[-1]

                # assert self._exists(
                #     model.Feature, name=feature
                # ), f"{feature} does not exist in the feature store"
                assert not self.list_features(name=feature_name, namespace=feature_namespace).empty, f"'{feature}' does not exist in the feature store"
            
            namespace = t_namespace or DEFAULT_NAMESPACE
            transform = {"function": func, "args": computed_from}
            payload = {"transform": transform, "description": func.__doc__}
            if not self.list_features(namespace=namespace, name=name).empty:
                # Already exists, update it
                self.update_feature(name, namespace=namespace, check=check, **payload)
            else:
                # Create a new feature
                self.create_feature(name, namespace=namespace, check=check, **payload)

            # Call the transform
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func

        return decorator


class InFeature(object):
    def __init__(self, name, namespace=None):
        self.name = name
        self.namespace = namespace
    
    def __str__(self):
        return f"{self.namespace}/{self.name}"


class OutFeature(object):
    def __init__(self, name, namespace=None):
        self.name = name
        self.namespace = namespace
    
    def __str__(self):
        return f"{self.namespace}/{self.name}"