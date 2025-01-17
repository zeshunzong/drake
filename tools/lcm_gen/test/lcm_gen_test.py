from pathlib import Path
import tempfile
import unittest

from bazel_tools.tools.python.runfiles import runfiles

from drake.tools.lcm_gen import (
    Parser,
    PrimitiveType,
    UserType,
)


class BaseTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self._manifest = runfiles.Create()

        self._lima_path = Path(self._manifest.Rlocation(
            "drake/tools/lcm_gen/test/lima.lcm"))
        assert self._lima_path.exists()

        self._mike_path = Path(self._manifest.Rlocation(
            "drake/tools/lcm_gen/test/mike.lcm"))
        assert self._mike_path.exists()


class TestParser(BaseTest):
    """Tests for the Parser class."""

    @staticmethod
    def _join_lines(lines):
        return "\n".join(lines) + "\n"

    def test_remove_c_comments(self):
        """C comments are replaced by whitespace (to preserve line numbers in
        error messages).
        """
        self.assertMultiLineEqual(
            Parser._remove_c_comments(self._join_lines([
                "foo",
                "bar /* comment",
                "still comment ",
                "comment */ eol",
                "bar/*quux*/baz",
            ])),
            self._join_lines([
                "foo",
                "bar           ",
                "              ",
                "           eol",
                "bar        baz",
            ]))

    def test_remove_cpp_comments(self):
        """C++ comments are snipped to end-of-line."""

        self.assertMultiLineEqual(
            Parser._remove_cpp_comments(self._join_lines([
                "foo",
                "bar // eol",
                "// line",
                "next",
            ])),
            self._join_lines([
                "foo",
                "bar ",
                "",
                "next"
            ]))

    def test_parse_lima(self):
        """Checks the parse tree for `lima.lcm`."""
        lima = Parser.parse(filename=self._lima_path)
        self.assertEqual(str(lima), """\
struct papa.lima {
  const double charlie_delta = 3.25e1;
  const float charlie_foxtrot = 4.5e2;
  const int8_t charlie_india8 = -8;
  const int16_t charlie_india16 = 16;
  const int32_t charlie_india32 = 32;
  const int64_t charlie_india64 = 64;
  boolean golf;
  byte bravo;
  double delta;
  float foxtrot;
  int8_t india8;
  int16_t india16;
  int32_t india32;
  int64_t india64;
}
""")
        # Check the value <=> value_str equivalence.
        for c in lima.constants:
            if c.typ in (PrimitiveType.float, PrimitiveType.double):
                self.assertIsInstance(c.value, float)
                self.assertEqual(c.value, float(c.value_str))
            else:
                self.assertIsInstance(c.value, int)
                self.assertEqual(c.value, int(c.value_str))

    def test_parse_mike(self):
        """Checks the parse tree for `mike.lcm`."""
        mike = Parser.parse(filename=self._mike_path)
        self.assertEqual(str(mike), """\
struct papa.mike {
  double delta[3];
  float foxtrot[4][5];
  papa.lima alpha;
  int32_t rows;
  int32_t cols;
  byte bravo[rows];
  int8_t india8[rows][cols];
  int16_t india16[7][cols];
  int32_t india32[rows][11];
  papa.lima xray[2];
  papa.lima yankee[rows];
  papa.lima zulu[rows][2];
}
""")
        # Check one field carefully for its exact representation.
        (zulu,) = [f for f in mike.fields if f.name == "zulu"]
        self.assertEqual(zulu.typ.package, "papa")
        self.assertEqual(zulu.typ.name, "lima")
        self.assertEqual(zulu.array_dims[0], "rows")
        self.assertEqual(zulu.array_dims[1], 2)

    def test_type_str(self):
        """Tests UserType.__str__. The end-to-end tests of the parser wouldn't
        usually hit these corner cases because most types are immediately
        resolved to use fully-qualified names (i.e., with a non-None package).
        """
        dut = UserType(package=None, name="bar")
        self.assertEqual(str(dut), "bar")
        dut = UserType(package="foo", name="bar")
        self.assertEqual(str(dut), "foo.bar")

    def _parse_str(self, content):
        with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="lcm_gen_test_",
                encoding="utf-8") as f:
            f.write(content)
            f.flush()
            return Parser.parse(filename=f.name)

    def test_missing_package_semi(self):
        with self.assertRaisesRegex(SyntaxError, "Expected ';'.*got.*struct"):
            self._parse_str('package foo /*;*/ struct empty { }')

    def test_multiline_const(self):
        foo = self._parse_str('struct foo { const int8_t x = 1, y = 2; }')
        self.assertEqual(foo.constants[0].name, "x")
        self.assertEqual(foo.constants[0].value, 1)
        self.assertEqual(foo.constants[1].name, "y")
        self.assertEqual(foo.constants[1].value, 2)

    def test_bad_const_type(self):
        with self.assertRaisesRegex(SyntaxError, "Expected.*primitive type"):
            self._parse_str('struct bad { const string name = "foo"; }')
        with self.assertRaisesRegex(SyntaxError, "Expected.*primitive type"):
            self._parse_str('struct bad { const bignum x = 2222; }')

    def test_bad_const_value(self):
        with self.assertRaisesRegex(SyntaxError, "Invalid.*value.*0x7f"):
            self._parse_str('struct bad { const int8_t x = 0x7f; }')

    def test_missing_const_name(self):
        with self.assertRaisesRegex(SyntaxError, "Expected.*NAME"):
            self._parse_str('struct bad { const double /* name */ = 1; }')
