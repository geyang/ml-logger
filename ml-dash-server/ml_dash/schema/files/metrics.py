from os.path import split, realpath, join
from graphene import relay, ObjectType, String, List, JSONString
from graphene.types.generic import GenericScalar
from ml_dash import schema
from ml_dash.config import Args
from ml_dash.schema.files.file_helpers import find_files, read_records, read_dataframe


class Metrics(ObjectType):
    class Meta:
        interfaces = relay.Node,

    keys = List(String, description="list of keys for the metrics")

    # value = List(GenericScalar, description="the raw value")
    value = GenericScalar(description="The value of the metrics file", keys=List(String))

    def resolve_keys(self, info):
        df = read_dataframe(join(Args.logdir, self.id[1:]))
        keys = df.dropna().keys()
        return list(keys)

    # add more complex queries.
    def resolve_value(self, info, keys=None):
        _ = read_dataframe(join(Args.logdir, self.id[1:]))
        if keys:
            df = _[keys].dropna()
            return {k: df[k].values.tolist() for k in keys}
        else:
            df = _.dropna()
            return {k: v.values.tolist() for k, v in df.items()}

    @classmethod
    def get_node(cls, info, id):
        return Metrics(id)


class MetricsConnection(relay.Connection):
    class Meta:
        node = Metrics


def get_metrics(id):
    # note: this is where the key finding happens.
    return Metrics(id=id)


def find_metrics(cwd, **kwargs):
    from ml_dash.config import Args
    _cwd = realpath(join(Args.logdir, cwd[1:]))
    parameter_files = find_files(_cwd, "**/metrics.pkl", **kwargs)
    for p in parameter_files:
        yield Metrics(id=join(cwd, p['path']))