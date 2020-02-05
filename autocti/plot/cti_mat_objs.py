from autoarray.plot import mat_objs


class ParallelOverscanLiner(mat_objs.Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(ParallelOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="parallel_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return ParallelOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialPrescanLiner(mat_objs.Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialPrescanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_prescan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialPrescanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialOverscanLiner(mat_objs.Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )
