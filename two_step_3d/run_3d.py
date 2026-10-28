"""Entrypoint for the 3D KPP experiment (Sec 4.4)."""

from .runner import build_parser, run


def main() -> None:
    args = build_parser().parse_args()
    out = run("kpp3d", args.mode, args)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
