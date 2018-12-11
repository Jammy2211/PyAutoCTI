from os import path

from autofit import conf

workspace_directory = "{}/../workspace".format(path.dirname(path.realpath(__file__)))

conf.instance = conf.Config("{}/config".format(workspace_directory), "{}/output/".format(workspace_directory))
