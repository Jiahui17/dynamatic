// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @extsu_32_64(
// CHECK-SAME:                           %[[IN:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, in>,
// CHECK-SAME:                           %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i64, out>) {

// CHECK:           next (%[[IN]]: !xls.schan<i32, in>, %[[OUT]]: !xls.schan<i64, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[DATA_IN:.*]] = xls.sblocking_receive %[[TOK0]], %[[IN]] : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:             %[[RESULT:.*]] = xls.zero_ext %[[DATA_IN]] : (i32) -> i64
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[RESULT]], %[[OUT]] : (!xls.token, i64, !xls.schan<i64, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_intcast(%in: !handshake.channel<i32>) -> (!handshake.channel<i64>) {
    %0 = extui %in : <i32> to <i64>
    end %0: <i64>
  }
}
