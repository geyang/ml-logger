from graphene import relay, ObjectType, Float, Schema, AbstractType, List, String, Union, Field, GlobalID, ID, Argument
from graphene.relay.node import from_global_id
from ml_dash.schema.files.series import Series, get_series
from ml_dash.schema.files.metrics import Metrics, get_metrics
from ml_dash.schema.schema_helpers import bind, bind_args
from ml_dash.schema.users import User, get_users, get_user
from ml_dash.schema.projects import Project
from ml_dash.schema.directories import Directory
from ml_dash.schema.files import File
from ml_dash.schema.experiments import Experiment


# class Experiment(graphene.ObjectType):
#     class Meta:
#         interfaces = relay.Node,
#
#     parameter_keys = graphene.List(description="keys in the parameter file")
#     metric_keys = graphene.List(description="the x data")
#     video_keys = graphene.List(description="the x data")
#     img_keys = graphene.List(description="the x data")
#     diff_keys = graphene.List(description="the x data")
#     log_keys = graphene.List(description="the x data")
#     view_config = ""
#
# class TimeSeries(graphene.ObjectType):
#     class Meta:
#         interfaces = relay.Node,
#
#     x_data = graphene.List(description="the x data")
#     y_data = graphene.List(description="the y data")
#     serialized = graphene.String(description='string serialized data')
#
#
# class TimeSeriesWithStd(graphene.ObjectType):
#     class Meta:
#         interfaces = relay.Node,
#
#     x_data = graphene.List(description="the x data")
#     y_data = graphene.List(description="the y data")
#     std_data = graphene.List(description="the standard deviation data")
#     quantile_25_data = graphene.List(description="the standard deviation data")
#     quantile_50_data = graphene.List(description="the standard deviation data")
#     quantile_75_data = graphene.List(description="the standard deviation data")
#     quantile_100_data = graphene.List(description="the standard deviation data")
#     mode_data = graphene.List(description="the standard deviation data")
#     mean_data = graphene.List(description="the standard deviation data")
#     serialized = graphene.String(description='string serialized data')
#
#
# class LineChart(graphene.ObjectType):
#     class Meta:
#         interfaces = relay.Node,
#
#     key = graphene.String(description="The path to the metrics file (including metrics.pkl)")
#     x_key = graphene.String(description="key for the x axis")
#     x_label = graphene.String(description="label for the x axis")
#     y_key = graphene.String(description="key for the y axis")
#     y_label = graphene.String(description="label for the x axis")


class EditText(relay.ClientIDMutation):
    class Input:
        text = String(required=True, description='updated content for the text file')

    text = String(description="the updated content for the text file")

    @classmethod
    def mutate_and_get_payload(cls, root, info, text, ):
        return dict(text=text)


class Query(ObjectType):
    node = relay.Node.Field()
    # context?
    # todo: files
    # todo: series

    users = Field(List(User), resolver=bind_args(get_users))
    user = Field(User, username=String(), resolver=bind_args(get_user))

    # Not Implemented atm
    # teams = relay.Node.Field(List(Team))
    # team = relay.Node.Field(Team)

    projects = relay.Node.Field(List(Project))
    project = relay.Node.Field(Project)

    metrics = Field(Metrics, id=Argument(ID))

    def resolve_metrics(self, info, id=None):
        _type, _id = from_global_id(id)
        return get_metrics(_id)

    series = Field(Series, prefix=String(), metrics_files=List(String),
                   window=Float(), x_key=String(), y_key=String(), label=String())

    def resolve_series(self, info, **kwargs):
        return get_series(**kwargs)


class Mutation(ObjectType):
    # todo: remove_file
    # todo: rename_file
    # todo: edit_file
    # todo: move_file
    # todo: copy_file

    update_text = EditText.Field()


schema = Schema(query=Query, mutation=Mutation)
